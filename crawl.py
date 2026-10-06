"""1단계: 패스트캠퍼스 강의 목록 수집 -> data/courses.json

- 사이트가 카테고리 화면에 쓰는 목록 데이터(/.api/categories/번호)를 카테고리별로 1번씩만 요청합니다.
  (강의 상세 페이지는 열지 않습니다. 전체 요청 약 10번)
- 공식 공개 API가 아니므로 예고 없이 바뀌거나 막힐 수 있습니다.
- 교육·시연 목적의 수집용입니다. 수집한 내용을 재배포하거나 영리 목적으로 쓰지 마세요.

사용법:  uv run python crawl.py            (전체)
         uv run python crawl.py --limit 100 (분야별로 고르게 100개만)
         uv run python crawl.py --categories BIZ,BIZ_PLANNING,PROGRAMMING (일부 분야만)
분야 코드: BIZ(AI/업무생산성) BIZ_PLANNING(비즈니스/기획) PROGRAMMING(개발/데이터) DATASCIENCEDL(AI TECH)
          FINANCE(금융/투자) DGN(디자인) VIDEO(영상/3D) ILLUST(드로잉/일러스트) AICREATIVE(AI CREATIVE)
실행할 때마다 목록을 통째로 새로 받아 덮어씁니다. (종료된 강의는 자동으로 빠짐)
"""
import argparse, json, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib import robotparser

import requests

BASE = "https://fastcampus.co.kr"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; course-rag-demo/1.0; educational use)",
    "Accept": "application/json",
    "Referer": BASE + "/",
}
OUT = Path(__file__).parent / "data" / "courses.json"
DELAY = 1.0
# 내비게이션에서 카테고리 번호를 못 읽었을 때 쓰는 기본값 (2026-10 기준 9개 카테고리)
FALLBACK_CATEGORY_IDS = [39, 921, 1, 10, 27, 28, 7, 116, 124]
SKIP_FORMATS = {"B2G_PROMOTION"}  # 검색 노출용 항목 등 실제 강의가 아닌 것


def get_json(session, path):
    r = session.get(BASE + path, timeout=30)
    r.raise_for_status()
    return r.json().get("data")


def category_ids(session):
    """상단 메뉴 데이터에서 카테고리 번호를 찾는다."""
    ids = []

    def walk(node):
        if isinstance(node, list):
            for n in node:
                walk(n)
        elif isinstance(node, dict):
            cid = node.get("categoryId")
            if cid and cid not in ids:
                ids.append(cid)
            for v in node.values():
                if isinstance(v, (list, dict)):
                    walk(v)

    try:
        walk(get_json(session, "/.api/operations/navigation?type=GLOBAL_NAVIGATION_COMMON"))
    except Exception as e:
        print(f"메뉴에서 카테고리를 읽지 못해 기본 목록을 씁니다: {e}")
    return ids or FALLBACK_CATEGORY_IDS


def hours_text(running_time):
    """'12:8'(시간:분) -> '약 12시간'. 값이 없거나 0이면 빈 문자열."""
    try:
        h, m = (int(x) for x in str(running_time).split(":")[:2])
    except Exception:
        return ""
    if h == 0 and m == 0:
        return ""
    return f"약 {h}시간" if h else f"약 {m}분"


def to_course(c, now):
    cat = (c.get("category") or {}).get("title") or ""
    sub = (c.get("subCategory") or {}).get("title") or ""
    fmt = (c.get("format") or {}).get("title") or ""
    badge = ((c.get("cardInfo") or {}).get("standardBadgeTitle") or "").strip()
    keywords = [k for k in (c.get("keywords") or []) if k]
    desc = " ".join((c.get("publicDescription") or "").split())
    level = c.get("qualification") or ""
    hours = hours_text(c.get("runningTime"))

    lines = [desc] if desc else []
    if keywords:
        lines.append("키워드: " + ", ".join(keywords))
    if level:
        lines.append("수강 대상: " + level)
    if hours:
        lines.append("수강 시간: " + hours)
    lines.append("분류: " + " > ".join(x for x in (cat, sub) if x))
    if badge or fmt:
        lines.append("형태: " + " / ".join(x for x in (badge, fmt) if x))

    return {
        "id": c.get("id"),
        "url": f"{BASE}/{c['slug']}",
        "title": (c.get("publicTitle") or "").strip(),
        "description": desc,
        "category": cat,
        "categories": [cat] if cat else [],
        "subcategory": sub,
        "type": badge or fmt or "온라인",
        "format": fmt,
        "level": level,
        "hours": hours,
        "keywords": keywords,
        "state": c.get("state") or "",
        "image": c.get("desktopCardAsset") or "",
        "text": "\n".join(lines),
        "crawled_at": now,
    }


def collect(session, ids, delay=DELAY, only=None):
    """only: 수집할 분야의 이름 또는 코드 목록 (예: ["BIZ", "비즈니스/기획"]). None 이면 전체."""
    want = {x.strip().lower() for x in only} if only else None
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    courses = {}
    for cid in ids:
        try:
            data = get_json(session, f"/.api/categories/{cid}") or {}
        except Exception as e:
            print(f"카테고리 {cid} 실패: {e}")
            continue
        title = data.get("title", cid)
        if want and not ({str(title).lower(), str(data.get("code", "")).lower()} & want):
            print(f"{title} [{data.get('code', '')}]: 건너뜀")
            time.sleep(delay)
            continue
        added = 0
        for c in data.get("courses") or []:
            if not c.get("slug") or not c.get("publicTitle"):
                continue
            if c.get("state") != "ONGOING":  # 준비 중(READY) 등은 제외
                continue
            if (c.get("format") or {}).get("code") in SKIP_FORMATS:
                continue
            if c["id"] in courses:  # 다른 카테고리에서 이미 받은 강의 -> 카테고리만 추가
                if title not in courses[c["id"]]["categories"]:
                    courses[c["id"]]["categories"].append(title)
                continue
            courses[c["id"]] = to_course(c, now)
            if title not in courses[c["id"]]["categories"]:
                courses[c["id"]]["categories"].append(title)
            added += 1
        print(f"{title} [{data.get('code', '')}]: 새 강의 {added}개 (누적 {len(courses)}개)")
        time.sleep(delay)
    return list(courses.values())


def balanced(courses, limit):
    """분야별로 돌아가며 하나씩 뽑아 limit 개를 고른다. (한 분야로 쏠리지 않게)"""
    groups = {}
    for c in courses:
        groups.setdefault(c["category"], []).append(c)
    picked, queues = [], list(groups.values())
    while len(picked) < limit and any(queues):
        for q in queues:
            if q and len(picked) < limit:
                picked.append(q.pop(0))
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="저장할 강의 수 (0=전체)")
    ap.add_argument("--categories", default="", help="수집할 분야만 쉼표로 구분 (이름 또는 코드). 예: BIZ,BIZ_PLANNING,PROGRAMMING")
    args = ap.parse_args()
    only = [x for x in args.categories.split(",") if x.strip()] or None

    rp = robotparser.RobotFileParser(f"{BASE}/robots.txt")
    rp.read()
    if not rp.can_fetch(HEADERS["User-Agent"], f"{BASE}/.api/categories/1"):
        raise SystemExit("robots.txt 가 이 경로의 수집을 허용하지 않습니다. 중단합니다.")

    s = requests.Session()
    s.headers.update(HEADERS)
    ids = category_ids(s)
    print(f"카테고리 {len(ids)}개 확인")
    courses = collect(s, ids, only=only)
    if not courses:
        raise SystemExit("강의를 하나도 받지 못했습니다. 기존 데이터는 그대로 둡니다.")
    total = len(courses)
    if args.limit:
        courses = balanced(courses, args.limit)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(courses, ensure_ascii=False, indent=1), encoding="utf-8")
    note = f" (전체 {total}개 중 분야별로 고르게 선택)" if len(courses) < total else ""
    print(f"완료: {len(courses)}개 강의{note} -> {OUT}")
    for cat, n in sorted(Counter(c["category"] for c in courses).items(), key=lambda x: -x[1]):
        print(f"  {cat}: {n}개")


if __name__ == "__main__":
    main()
