"""Create 400 distinct, stratified cases plus disjoint pilot cases.

Source-document labels are silver known-item labels, NOT expert relevance gold.
Question text and evidence stay in the operator's private output directory.
"""
from __future__ import annotations
import argparse
import asyncio
from collections import defaultdict, Counter
import hashlib
import json
import logging
from pathlib import Path
import random
import re

from moleg_paper_runtime import bootstrap


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    _, serving, config, resolve = bootstrap()
    from infra.vector_db.vector_store import VectorStore
    from config import settings
    from openai import AsyncOpenAI
    logging.disable(logging.CRITICAL)
    vs = VectorStore()
    result = vs.client.search(index=settings.INDEX_NAME_ARTICLE, body={
        "size": 8000, "_source": {"excludes": ["embedding_full_text", "embedding_body_text"]},
        "query": {"function_score": {"random_score": {"seed": 20260905, "field": "_seq_no"}}}})
    groups = defaultdict(list)
    seen_laws = set()
    for hit in result["hits"]["hits"]:
        s = hit["_source"]
        body = str(s.get("paragraph_content") or s.get("body_text") or "")
        title = str(s.get("title") or "")
        lid = str(s.get("law_id") or str(s.get("item_id", hit["_id"])).split("_")[0])
        if (lid in seen_laws or len(body) < 100 or len(body) > 5000 or "폐지" in title
                or not s.get("country") or not title
                or len(re.findall("[가-힣]", body)) / max(1, len(body)) < .3):
            continue
        seen_laws.add(lid)
        groups[s["country"]].append({"source_id": str(s.get("item_id") or hit["_id"]),
            "source_law_id": lid, "country": s["country"], "title": title,
            "subject": s.get("subject_title") or "", "evidence": body,
            "source_index": settings.INDEX_NAME_ARTICLE})
    docs = []
    while groups and len(docs) < 280:
        for country in sorted(list(groups)):
            docs.append(groups[country].pop(0))
            if not groups[country]:
                del groups[country]
    if len(docs) < 264:
        raise RuntimeError(f"Only {len(docs)} eligible distinct source laws")
    model_cfg = next(x["litellm_params"] for x in config["model_list"] if x["model_name"] == "worker agent")
    cli = AsyncOpenAI(base_url=resolve(model_cfg["api_base"]),
        api_key=resolve(model_cfg.get("api_key")) or serving["VLLM_API_KEY"], timeout=90, max_retries=0)
    sem = asyncio.Semaphore(4)
    args.output.mkdir(parents=True, exist_ok=True)
    draft_path = args.output / 'question_drafts.jsonl'
    drafts = {}
    if draft_path.exists():
        for line in draft_path.read_text().splitlines():
            value = json.loads(line)
            drafts[value['source_id']] = value
    draft_file = draft_path.open('a', encoding='utf-8', buffering=1)

    async def make(i, doc):
        if doc['source_id'] in drafts:
            return drafts[doc['source_id']]
        kind = ["named_law", "article_subject", "source_paraphrase"][i % 3]
        out = {**doc, "kind": kind, "label_type": "source_derived_silver",
               "expected_task": "law_search", "expert_verified": False}
        if kind == "named_law":
            short = re.split(r"[（(]", doc["title"], maxsplit=1)[0].strip()
            out["question"] = f"{doc['country']}의 「{short}」 법령과 주요 조문을 찾아주세요."
            out["reference_answer"] = ""
            draft_file.write(json.dumps(out, ensure_ascii=False)+'\n')
            return out
        prompt = ("주어진 법령 발췌만으로 답할 수 있는 한국어 검색 질문 하나를 JSON으로 작성하세요. "
                  "question은 국가명을 포함하고 자기완결적이어야 합니다. '아래/이 조문/주어진'은 쓰지 마세요. "
                  "질문에 정답 내용을 넣지 말고, 발췌에 없는 의무/조건을 만들지 마세요. "
                  "reference_answer는 질문의 답이 되는 발췌 원문 연속 구절을 30~250자로 그대로 복사하세요. "
                  + ("법령명과 조문 주제를 명시해 특정 조문을 찾는 질문을 쓰세요." if kind == "article_subject"
                     else "법령명/조 번호를 쓰지 말고, 실제 상황에서 해당 규정을 찾는 자연스러운 질문을 쓰세요.")
                  + '\n출력: {"question":"...", "reference_answer":"..."}')
        async with sem:
            r = await cli.chat.completions.create(model=model_cfg["model"][len("openai/"):],
                messages=[{"role":"system","content":prompt}, {"role":"user","content":json.dumps(doc, ensure_ascii=False)}],
                temperature=0, max_tokens=1200, response_format={"type":"json_object"},
                extra_body={"reasoning_effort":"low"})
        try:
            value = json.loads(r.choices[0].message.content)
            question, reference = value["question"].strip(), value["reference_answer"].strip()
        except (ValueError, KeyError, TypeError):
            question, reference = '', ''
        out['question_generation_finish_reason'] = r.choices[0].finish_reason
        # Fail closed on invented answer spans. Failed drafts are retained in the
        # manifest and replaced with a deterministic subject lookup (labelled).
        if len(reference) < 20 or reference not in doc["evidence"] or len(question) < 15:
            out["kind"] = "article_subject_fallback"
            out["generation_validation_failed"] = True
            question = f"{doc['country']} 「{doc['title']}」의 {doc['subject']} 관련 조문을 찾아주세요."
            reference = ""
        if doc["country"] not in question:
            question = doc["country"] + "에서 " + question
        out["question"], out["reference_answer"] = question, reference
        draft_file.write(json.dumps(out, ensure_ascii=False)+'\n')
        print('question', i, out['kind'], flush=True)
        return out

    sourced = await asyncio.gather(*(make(i,d) for i,d in enumerate(docs[:264])))
    draft_file.close()
    cases, pilot = sourced[:240], sourced[240:264]
    countries = ["미국","독일","일본","프랑스","중국","싱가포르","베트남","호주","영국","캐나다"]
    scenarios = ["직원을 채용할 때 근로계약에 포함할 사항", "직원 해고 전에 지켜야 하는 절차",
        "온라인 쇼핑몰 소비자의 청약 철회", "고객 개인정보를 해외 서버로 이전하는 조건",
        "외국인이 현지 회사를 설립할 때의 요건", "직장 내 성희롱 예방 의무",
        "제품 결함으로 소비자가 피해를 입었을 때의 책임", "전자서명의 효력",
        "아동의 온라인 개인정보 수집", "저작권자의 허락 없이 콘텐츠를 이용할 수 있는 예외"]
    for c in countries:
        for topic in scenarios:
            cases.append({"question":f"{c}에서 {topic}에 관한 법령과 조문을 찾아주세요.",
                "kind":"scenario", "country":c, "label_type":"unlabelled", "expected_task":"law_search"})
    for i in range(20):
        a,b = countries[i%10], countries[(i+3)%10]
        topic = ["개인정보 보호", "근로계약"][i//10]
        cases.append({"question":f"{a}과 {b}의 {topic} 관련 법령을 비교하고 공통점과 차이를 설명해주세요.",
            "kind":"comparison", "label_type":"unlabelled", "expected_task":"law_search"})
    snippets = ["당사자는 서면으로 계약을 해지할 수 있다.", "개인정보는 정해진 목적에 한하여 처리한다.",
        "근로자는 안전한 작업 환경을 제공받을 권리가 있다.", "소비자는 계약 체결 전에 가격을 고지받아야 한다.",
        "이 법은 공포한 날부터 시행한다.", "미성년자는 법정대리인의 동의를 받아야 한다.",
        "신청인은 처분에 대하여 이의를 제기할 수 있다.", "사업자는 기록을 3년간 보관한다.",
        "계약 당사자는 비밀유지 의무를 부담한다.", "위반자는 손해를 배상하여야 한다."]
    for language in ["영어", "일본어"]:
        for snippet in snippets:
            cases.append({"question":f"다음 법률 문장을 {language}로 번역해주세요: {snippet}",
                "kind":"assistant_translation", "label_type":"unlabelled", "expected_task":"assistant"})
    chats = ["안녕하세요", "안녕!", "반갑습니다", "hello", "hi!", "감사합니다", "고마워요", "정말 감사합니다",
        "도움이 됐어요, 고맙습니다", "알려줘서 고마워요", "이 시스템에서 무엇을 할 수 있나요?",
        "검색 기능 사용법을 알려주세요", "서비스 이용 안내를 보고 싶어요", "어떤 국가의 법령을 검색할 수 있나요?",
        "검색 결과는 어떻게 읽으면 되나요?", "오늘 날씨를 알려줘", "맛있는 김치찌개 만드는 법 알려줘",
        "파이썬 리스트 정렬 방법을 알려줘", "좋아하는 음악이 뭐야?", "이번 주말 여행지를 추천해줘"]
    for q in chats:
        cases.append({"question":q, "kind":"conversation", "label_type":"unlabelled"})
    assert len(cases) == 400 and len({x["question"] for x in cases}) == 400
    for i,c in enumerate(cases): c["case_id"] = f"q{i:03d}"
    for i,c in enumerate(pilot): c["case_id"] = f"pilot{i:02d}"
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in [("cases.json", cases), ("pilot.json", pilot)]:
        (args.output/name).write_text(json.dumps(rows, ensure_ascii=False, indent=2)+"\n")
    manifest = {"seed":20260905, "n":400, "unique_questions":400,
        "kinds":dict(Counter(c["kind"] for c in cases)), "source_laws":240,
        "source_countries":dict(Counter(c["country"] for c in cases if c.get("source_id"))),
        "expert_gold":False, "pilot_disjoint":True,
        "cases_sha256":hashlib.sha256((args.output/'cases.json').read_bytes()).hexdigest(),
        "counts":{index:vs.client.count(index=index)["count"] for index in
                  [settings.INDEX_NAME_ARTICLE, settings.INDEX_NAME_WORLD_LAW_ORIGIN]},
        "source_query":"random_score seed=20260905, field=_seq_no; country round-robin; distinct law_id"}
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(manifest,ensure_ascii=False),flush=True)
    await cli.close()


if __name__ == "__main__": asyncio.run(main())
