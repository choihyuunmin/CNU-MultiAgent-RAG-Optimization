# CNU 추론 구간 계측·조회 겹치기 어댑터

상태: 2026-09-11 구현·로컬 계약 시험 완료. 실제 검색기 배포와 신규 GPU 속도·정확도 검증은 미실행. 이전의 고정 호출 허가 연구를 대체하는 현재 실행 경로다. 과거 정책 코드는 재현 목적으로 남기지만 현재 진입점에서 사용하지 않는다.

## 무엇을 제거했는가

`scripts/serve_embedded_adapter.py`에서 NetworkHarness·CompletionRouter·슬롯 획득·질문 입구 교체·전달 지연 제거를 모두 제거했다. 원래 비동기 함수에 메타데이터 관측만 붙인다. `delivery`와 `network` 모드는 더 이상 허용하지 않는다. 명시한 원래 함수가 없으면 연결을 거부한다. 세마포어·호출 수 제한·재시도·추론 중단·모델 변경·API 인자 변경은 새 어댑터에 없다.

이 변경은 어댑터가 추가한 제한을 제거하는 것이다. 원래 검색기의 사용자 세션 직렬화·DB 풀·입구 안전 제한까지 무조건 삭제하지 않는다. 기존 실험용 스케줄러 모듈과 기록은 보존하며, 원격 운영 프로세스는 이번 작업에서 수정하지 않았다.

## 실제 로그에서 확인한 병목과 미확인 원인

9월 1일 실행의 로컬 보관 원시 `LLM timing` 로그에서 확인한 두 호출이다. 아래는 시간 필드만 발췌한 값이며 신규 실험이나 CoT 본문이 아니다.

| 관측 항목 | 호출 A | 호출 B |
|---|---:|---:|
| 모델 호출 전체 | 38,081.5ms | 40,776.9ms |
| 첫 청크 | 107.8ms | 130.6ms |
| 첫 가시 출력 | 37,341.4ms | 39,179.9ms |
| 첫·마지막 reasoning 청크 사이 | 37,219.8ms | 39,018.3ms |

이 두 사례는 첫 청크 전 연결·서빙 대기보다 가시 출력 전 추론 구간이 길다. 그러나 client 청크 간 시간은 순수 GPU 계산 시간이 아니며 네트워크·소비자 정지도 섞일 수 있다. 전체 100문항의 대표성이나 원인을 이 두 사례로 증명하지 않는다.

CoT 원문은 `raw_text_logged=0`으로 보존되지 않았다. 따라서 “동일 판단을 반복했다”, “특정 문장을 없앴다”는 결론을 내릴 수 없다. 공급자 reasoning_tokens가 completion_tokens보다 큰 집계도 관측되어 토큰 비율은 확정하지 않았다. 새 관측기는 이 경우 `usage_inconsistent=true`를 표시한다. 비스트리밍 응답에는 토큰 수만 기록하고 추론 시간을 추정하지 않는다.

추론 내용·모델·호출 조건을 그대로 유지하면서 외부 코드가 내부 autoregressive 연산을 건너뛸 수 있다는 근거는 없다. 이 구현을 CoT 단축이나 새 디코딩 알고리즘으로 설명하지 않는다.

## 핵심 기능: 같은 스냅샷의 독립 조회를 추론과 겹치기

기존: `모델 호출 완료 → 독립 메타데이터 조회 → 결합`

후보: `모델 호출과 독립 조회를 동시에 진행 → 실제 계약 확인 → 원래 방식으로 결합`

이상적인 구간 시간은 `L + R`에서 `max(L, R)`로 줄어든다. L은 변경하지 않은 모델 호출, R은 독립 조회다. 절약 상한은 `min(L, R)`이며 이는 상한 설명이지 실측 가속률이 아니다. 예를 들어 추론이 39초이고 조회가 0.6초면 이 방법으로 39초를 없앨 수 없다. 기존 경로가 이미 두 작업을 겹친다면 추가 이득도 없다.

단순 병렬화 함수와 달리 사용 지점에서 아래 계약 전체가 같아야 선행 결과를 채택한다.

- 실제 조회 입력의 정확한 바이트열: ID 순서·정렬·필드·언어·기본값 등 모든 유효 인자 포함.
- 조회 구현 버전과 작업 식별자.
- 사용자/테넌트 권한 범위 및 권한 버전. 비밀키를 식별자로 사용하지 않는다.
- 실제로 고정된 불변 데이터 리비전 또는 읽기 스냅샷.
- 읽기 전용이고 결정적이라는 통합 측의 명시적 계약.

현재 시각이나 임의 문자열을 snapshot으로 넣으면 검증이 아니다. 같은 snapshot에서 결과가 달라지는 검색, 공유 비동기 DB 세션, LLM 호출, 쓰기 작업에는 적용하지 않는다. 플래그만으로 순수성이나 실제 스냅샷 강제를 자동 증명하지 못한다.

입력·권한·버전 불일치 또는 선행 조회 오류면 결과를 버리고 원래 조회 함수를 실행한다. 요청 취소에서는 진행 조회를 취소·회수하고 새 fallback 요청을 만들지 않는다. 요청 사이 결과를 공유하거나 저장하지 않으며 한 결과는 한 번만 소비한다. 조회 함수는 기존 timeout과 취소 전파를 지켜야 한다. 선행 조회가 취소를 무시하면 정리도 지연될 수 있다.

## 적용 코드

```python
from cnu_rag_optimization.inference_overlap import (
    InferenceOverlapAdapter, ReadContract,
)

adapter = InferenceOverlapAdapter(on_event=write_metadata_event)
llm = adapter.completion(original_llm)  # 호출 인자와 응답 객체 유지

# 실제 입력이 완성되고, 해당 조회가 모델 출력에 의존하지 않을 때만 시작.
before = ReadContract(
    operation="document_metadata", implementation="reader-v1",
    authorization=verified_access_version, snapshot=pinned_revision,
    arguments=exact_read_arguments, read_only=True, deterministic=True,
)
async with adapter.prepare_read(before, original_metadata_read) as prepared:
    answer = await llm(**original_llm_arguments)
    actual = contract_from_authoritative_state()  # 현재 유효한 입력/권한/버전
    metadata = await prepared.consume(actual, original_metadata_read)
    result = original_merge(answer, metadata)
```

`stream()`은 기존 async generator 함수에 사용한다. coroutine이 SDK 스트림 객체를 반환하는 함수에 그대로 붙이지 않는다. 스트리밍 모드나 `include_usage` 옵션을 추가하지 않으며 이미 제공된 정보만 관측한다. 가시 출력 시점은 content 또는 tool_calls의 첫 관측이며, 여러 choice가 있으면 추론 span은 관측 전체 범위다. CoT·프롬프트·문서·응답 원문은 로그에 저장하지 않는다.

## 검색기 실행 접점

`serve_embedded_adapter.py`는 소스 지문을 확인한 뒤 원래 호출 함수 두 개를 공통 별칭에 연결한다. 예를 들어 원래 구현을 `_acompletion_via_proxy_unadmitted` 및 `_acompletion_stream_via_proxy_unadmitted`로 보존한 애플리케이션에서는 이 이름을 `--completion-source`, `--stream-source`에 명시한다. 실제 함수가 동일 모델 요청을 생성하는지는 배포 전 통합 테스트로 확인해야 한다. 직접 HTTP·다른 서비스의 호출까지 자동 연결되는 것은 아니다.

실행기는 계측만 연결한다. 도메인 코드를 모르는 상태에서 임의 조회를 선행 실행하지 않는다. 조회 겹치기는 위 코드처럼 검색기의 의존성·스냅샷 접점에 명시적으로 붙여야 한다. 기존 단순 병렬 보강이 이미 활성화된 경로에는 중복 적용하지 않는다. 원래 서비스 소스·설정·질문·주소·인증정보는 이 저장소에 포함하지 않는다.

## 검증과 한계

- 로컬 테스트: `python -m pytest -q`.
- 커밋 대상만 내보낸 깨끗한 사본에서 141개 통과. 기존 미커밋 시험을 포함한 작업 디렉터리에서는 152개 통과했다.
- 모의 동작 시험: `python examples/inference_overlap.py`.
- 150ms 모의 모델 + 60ms 모의 조회, 방법별 6회 교차 실행에서 약 213.22ms → 151.81ms. fixture 결과 동일, 모델 호출·조회 횟수 각각 12회. 실제 모델 추론이나 법령 정확도를 측정한 값이 아니다.
- 32개 호출이 추가 허가 없이 모두 원래 함수에 진입하는 시험 포함.
- 입력·반환 객체·스트림 청크·원래 예외 보존, 다른 권한/스냅샷/순서 fallback, 오류·취소 정리 시험 포함.

실제 개선 주장 전에는 같은 질문 100개에 기존 방식과 후보를 교차 반복하고 전체 지연·추론 span·잔여 조회 대기·폐기/중복 조회 수·자원 사용·기준 응답 재현율을 함께 측정해야 한다. 전문가 정답 없이 정확도 유지를 주장하지 않는다. 효과가 작으면 그대로 보고하고, 추론 자체를 줄였다고 이름을 바꾸지 않는다.

일반 비동기 작업과 취소 수명 관리는 Python 공식 문서의 `create_task`와 취소 전파 원칙을 따른다: https://docs.python.org/3/library/asyncio-task.html . 이는 독립 작업을 겹칠 수 있다는 구현 근거이지 새로운 논문 기여나 실제 속도 향상의 증거가 아니다.
