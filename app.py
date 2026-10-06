"""3단계: 챗봇 화면.  실행:  streamlit run app.py"""
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
import rag  # noqa: E402

LOG = ROOT / "logs" / "questions.jsonl"

st.set_page_config(page_title="패스트캠퍼스 강의 추천 챗봇", page_icon="🎓")
st.title("🎓 강의 추천 챗봇")
st.caption("패스트캠퍼스 공개 강의 소개를 검색해 추천하는 RAG 시연용 챗봇 (비공식)")

if not (rag.DATA / "courses.json").exists():
    st.warning("강의 데이터가 없습니다. 먼저 `crawl_class.bat` 또는 `uv run python crawl.py` 를 실행하세요.")
    st.stop()


@st.cache_resource(show_spinner="강의 색인을 만드는 중… (처음 한 번만 몇 분 걸릴 수 있어요)")
def load_index():
    return rag.CourseIndex()


index = load_index()
courses = index.courses

# ── 기준일 표시 ────────────────────────────────────
y, m, d = rag.data_asof(courses).split("-")
st.info(f"**{y}년 {int(m)}월 {int(d)}일** 수집 자료 기준 · 강의 {len(courses)}개. 이후 바뀐 내용은 반영되지 않았습니다.", icon="📅")


def options(key):
    """데이터에 실제로 있는 값만 선택지로 보여준다."""
    vals = set()
    for c in courses:
        v = c.get(key)
        vals.update(v if isinstance(v, list) else [v])
    return sorted(x for x in vals if x)


TIME_CHOICES = {"전체": None, "5시간 이하": (0, 5), "10시간 이하": (0, 10), "20시간 이하": (0, 20), "20시간 초과": (20, 10**6)}

with st.sidebar:
    # ── 조건 필터 ──────────────────────────────────
    st.subheader("조건 필터")
    f_cat = st.multiselect("분야", options("categories") or options("category"))
    f_sub = st.multiselect("세부 분류", options("subcategory"))
    f_level = st.multiselect("수강 대상", options("level"))
    f_time = st.selectbox("수강 시간", list(TIME_CHOICES), help="시간을 고르면 수강 시간 정보가 없는 강의는 제외됩니다.")

    def passes(c):
        if f_cat and not (set(c.get("categories") or [c.get("category")]) & set(f_cat)):
            return False
        if f_sub and c.get("subcategory") not in f_sub:
            return False
        if f_level and c.get("level") not in f_level:
            return False
        if TIME_CHOICES[f_time]:
            lo, hi = TIME_CHOICES[f_time]
            h = rag.hours_number(c)
            if h is None or not (lo < h <= hi if lo else h <= hi):
                return False
        return True

    filtering = bool(f_cat or f_sub or f_level or TIME_CHOICES[f_time])
    allowed = [i for i, c in enumerate(courses) if passes(c)] if filtering else None
    if filtering:
        st.caption(f"조건에 맞는 강의 **{len(allowed)}개** / 전체 {len(courses)}개")

    st.subheader("설정")
    k = st.slider("검색할 강의 수", 3, 8, 5)
    show = st.toggle("검색 근거 보기 (RAG의 R)", value=True)
    save_log = st.toggle("질문 기록 저장", value=True, help="질문, 검색된 강의, 답변을 logs/questions.jsonl 에 저장합니다.")
    st.markdown(f"- 강의 **{len(courses)}개** / 조각 **{len(index.chunks)}개**\n- 검색 방식: **{index.mode}**")
    if index.embed_stats:
        st.caption("임베딩: 재사용 {reused}개 · 새로 계산 {computed}개 · 삭제 {removed}개".format(**index.embed_stats))
    if st.button("대화 지우기"):
        st.session_state.msgs = []


def show_cards(cards):
    """추천 카드: 제목·링크·수강 대상·시간은 수집 데이터에서 그대로 가져온다 (AI가 쓴 글이 아님)."""
    if not cards:
        return
    st.caption("추천 강의 바로가기")
    for c in cards:
        with st.container(border=True):
            left, right = st.columns([1, 3])
            if c.get("image"):
                left.image(c["image"])
            meta = " · ".join(x for x in [c.get("category"), c.get("subcategory"), c.get("level"), c.get("hours")] if x)
            right.markdown(f"**{c['title']}**")
            right.caption(meta)
            if c.get("description"):
                right.write(c["description"][:120] + ("…" if len(c["description"]) > 120 else ""))
            right.link_button("강의 페이지 열기", c["url"])


def pick_cards(reply, results):
    """답변에 링크가 실린 강의만 카드로. 하나도 없으면(키 없음 등) 검색된 강의 전부."""
    named = [r for r in results if r["url"] in reply]
    keep = ["title", "url", "category", "subcategory", "level", "hours", "description", "image"]
    return [{k: r.get(k, "") for k in keep} for r in (named or results)]


def write_log(question, results, reply, filters):
    LOG.parent.mkdir(exist_ok=True)
    row = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "filters": filters,
        "results": [{"title": r["title"], "url": r["url"], "score": round(r["score"], 3)} for r in results],
        "answer": reply,
        "search_mode": index.mode,
    }
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


st.session_state.setdefault("msgs", [])
for msg in st.session_state.msgs:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_cards(msg.get("cards"))

if q := st.chat_input("예: 엑셀 반복 업무를 AI로 자동화하고 싶은 비개발자에게 맞는 강의는?"):
    st.chat_message("user").markdown(q)
    # 후속 질문("그 중 더 쉬운 건?")도 검색되도록 직전 질문을 함께 검색어로 사용
    prev = [m["content"] for m in st.session_state.msgs if m["role"] == "user"][-1:]
    results = index.search(" ".join(prev + [q]), k=k, allowed=allowed)

    with st.chat_message("assistant"):
        if show:
            with st.expander(f"🔎 검색된 강의 {len(results)}개"):
                if not results:
                    st.write("조건에 맞는 강의가 없습니다.")
                for r in results:
                    st.markdown(f"**[{r['title']}]({r['url']})** · {r['category']} · 유사도 {r['score']:.2f}")
                    st.caption(r["snippets"][0][:200] + "…")
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.msgs[-6:]]
        stream = rag.answer(q, results, history=history) if results else None
        if not results:
            reply = "선택한 조건에 맞는 강의가 없습니다. 왼쪽의 조건 필터를 넓혀서 다시 질문해 보세요."
            st.markdown(reply)
        elif stream is None:
            reply = "API 키가 없어 검색 결과만 보여드립니다 (`.env` 에 키를 넣으면 추천 설명이 생성됩니다).\n\n" + "\n".join(
                f"{n}. [{r['title']}]({r['url']}) — {r['description']}" for n, r in enumerate(results, 1))
            st.markdown(reply)
        else:
            reply = st.write_stream(stream)
        cards = pick_cards(reply, results) if results else []
        show_cards(cards)

    if save_log:
        try:
            write_log(q, results, reply, {"분야": f_cat, "세부 분류": f_sub, "수강 대상": f_level, "수강 시간": f_time})
        except Exception as e:  # 기록 실패가 답변을 막지 않게
            st.caption(f"질문 기록 저장 실패: {e}")

    st.session_state.msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": reply, "cards": cards}]

st.caption("질문과 답변은 품질 개선을 위해 이 컴퓨터의 `logs` 폴더에 저장됩니다. 사이드바에서 끌 수 있습니다. 개인정보는 입력하지 마세요.")
