"""Merge sweep parts into one layout: part1 (levels 8,16, 32 round-1), part2 (level 32 as round-2), part3 (64,100).
Copies client records, metrics and protocol/status; concatenates application logs per arm."""
import shutil, sys, json
from pathlib import Path
src = Path(sys.argv[1]); dst = Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=True)
main = dst/"main"; main.mkdir(exist_ok=True)
def copy_tree(s, d):
    if d.exists(): shutil.rmtree(d)
    shutil.copytree(s, d)
p1 = src/"part1"/"main"
for lvl in ["level-008", "level-016"]:
    copy_tree(p1/lvl, main/lvl)
(main/"level-032").mkdir(exist_ok=True)
copy_tree(p1/"level-032"/"round-1", main/"level-032"/"round-1")
p2 = src/"part2"/"main-part2"
if (p2/"level-032"/"round-1").exists():
    copy_tree(p2/"level-032"/"round-1", main/"level-032"/"round-2")
p3 = src/"part3"/"main-part3"
for lvl in ["level-064", "level-100"]:
    if (p3/lvl).exists(): copy_tree(p3/lvl, main/lvl)
# protocol/status per part
for name, p in [("part1", p1), ("part2", p2), ("part3", p3)]:
    for f in ["protocol.json", "status.json"]:
        if (p/f).exists(): shutil.copy(p/f, main/f"{f[:-5]}-{name}.json")
# logs: concatenate part1 + part2 (same pods; overlap is harmless for time-window assignment)
for arm in ["original", "capability", "direct"]:
    parts = [src/"part1"/f"{arm}-application.log", src/"part2"/f"{arm}-application.part2.log"]
    with (dst/f"{arm}-application.log").open("wb") as out:
        for p in parts:
            if p.exists(): out.write(p.read_bytes())
completed = []
for f in sorted(main.glob("status-*.json")):
    completed += json.load(open(f))["completed"]
print("cells:", [(c["level"], c["round"], c["method"], c["transport_ok"]) for c in completed])
print("levels present:", sorted(p.name for p in main.glob("level-*")))
