"""3단계: 챗봇 화면.  실행:  streamlit run app.py

화면 구성: 한 줄 결론 -> 추천 카드(가로) -> 이어서 물어보기 -> 평가.
검색 근거와 색인 정보는 사이드바의 '운영자 보기'를 켰을 때만 보인다.
"""
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

import common

ROOT = Path(__file__).parent
LOG = ROOT / "logs" / "questions.jsonl"
FEEDBACK = ROOT / "logs" / "feedback.jsonl"

st.set_page_config(page_title="패스트캠퍼스 강의 추천 챗봇", page_icon="🎓", layout="wide")
common.setup()  # .env / 배포 Secrets 읽기, 접속 암호 확인
import rag  # noqa: E402

if not (rag.DATA / "courses.json").exists():
    st.warning("강의 데이터가 없습니다. 먼저 `crawl_class.bat` 또는 `uv run python crawl.py` 를 실행하세요.")
    st.stop()


@st.cache_resource(show_spinner="강의 색인을 만드는 중… (처음 한 번만 몇 분 걸릴 수 있어요)")
def load_index():
    return rag.CourseIndex()


index = load_index()
courses = index.courses
ss = st.session_state
ss.setdefault("msgs", [])
ss.setdefault("pending", None)   # 버튼으로 넣은 질문
ss.setdefault("exclude", [])     # '다른 강의 더 보기'로 제외한 강의 주소

EXAMPLES = [
    "엑셀 반복 업무를 자동화하고 싶은 비개발자",
    "보고서와 기획서를 AI로 빨리 쓰고 싶어요",
    "데이터 분석을 처음 배우는 사람에게 맞는 강의",
    "ChatGPT를 업무에 제대로 쓰는 법",
]
TIME_CHOICES = {"전체": None, "5시간 이하": (0, 5), "10시간 이하": (0, 10), "20시간 이하": (0, 20), "20시간 초과": (20, 10**6)}
FILTERS = {"f_cat": ("분야", []), "f_sub": ("세부 분류", []), "f_level": ("수강 대상", []), "f_time": ("수강 시간", "전체")}
for key, (_, empty) in FILTERS.items():
    ss.setdefault(key, empty)


def options(key):
    """데이터에 실제로 있는 값만 선택지로 보여준다."""
    vals = set()
    for c in courses:
        v = c.get(key)
        vals.update(v if isinstance(v, list) else [v])
    return sorted(x for x in vals if x)


LEVELS = options("level")


def passes(c, f):
    if f["f_cat"] and not (set(c.get("categories") or [c.get("category")]) & set(f["f_cat"])):
        return False
    if f["f_sub"] and c.get("subcategory") not in f["f_sub"]:
        return False
    if f["f_level"] and c.get("level") not in f["f_level"]:
        return False
    if TIME_CHOICES[f["f_time"]]:
        lo, hi = TIME_CHOICES[f["f_time"]]
        h = rag.hours_number(c)
        if h is None or h > hi or (lo and h <= lo):
            return False
    return True


def current_filters():
    return {k: ss[k] for k in FILTERS}


def allowed_ids(f, exclude=()):
    active = any(f[k] != FILTERS[k][1] for k in FILTERS)
    if not active and not exclude:
        return None
    return [i for i, c in enumerate(courses) if passes(c, f) and c["url"] not in exclude]


# ── 버튼이 누르는 동작들 (화면을 다시 그리기 전에 상태를 바꾼다) ─────────────────
def last_question():
    users = [m["content"] for m in ss.msgs if m["role"] == "user"]
    return users[-1] if users else None


def ask(question, keep_exclude=False):
    if not keep_exclude:
        ss.exclude = []
    ss.pending = question


def clear_filter(key, reask=False):
    ss[key] = FILTERS[key][1]
    if reask and last_question():
        ask(last_question())


def clear_all_filters():
    for key, (_, empty) in FILTERS.items():
        ss[key] = empty


def narrow(key, value):
    """조건을 하나 걸고 직전 질문을 다시 검색"""
    ss[key] = value
    ask(last_question())


def more_courses(shown_urls):
    """이미 보여준 강의를 빼고 직전 질문을 다시 검색"""
    ss.exclude = list(dict.fromkeys(ss.exclude + shown_urls))
    ask(last_question(), keep_exclude=True)


def save_feedback(msg_id):
    value = ss.get(f"fb_{msg_id}")
    if value is None:
        return
    try:
        FEEDBACK.parent.mkdir(exist_ok=True)
        with FEEDBACK.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": msg_id, "time": datetime.now().isoformat(timespec="seconds"), "helpful": bool(value)}) + "\n")
    except Exception:
        pass


# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("조건 필터")
    st.multiselect("분야", options("categories") or options("category"), key="f_cat", placeholder="전체")
    st.multiselect("세부 분류", options("subcategory"), key="f_sub", placeholder="전체")
    st.multiselect("수강 대상", LEVELS, key="f_level", placeholder="전체")
    st.selectbox("수강 시간", list(TIME_CHOICES), key="f_time", help="시간을 고르면 수강 시간 정보가 없는 강의는 제외됩니다.")

    st.divider()
    operator = st.toggle("운영자 보기", value=False, help="검색 근거, 유사도, 색인 정보를 함께 보여줍니다. 수업 시연이나 점검용입니다.")
    save_log = st.toggle("질문 기록 저장", value=True, help="질문, 검색된 강의, 추천 결과를 logs 폴더에 저장합니다.")
    if operator:
        k = st.slider("검색할 강의 수", 3, 8, 5)
        st.markdown(f"- 강의 **{len(courses)}개** / 조각 **{len(index.chunks)}개**\n- 검색 방식: **{index.mode}**")
        if index.embed_stats:
            st.caption("임베딩: 재사용 {reused}개 · 새로 계산 {computed}개 · 삭제 {removed}개".format(**index.embed_stats))
    else:
        k = 5
    if st.button("대화 지우기", width="stretch"):
        ss.msgs, ss.exclude = [], []
        st.rerun()

# ── 머리말: 제목 + 기준일 ─────────────────────────────────────────────────────
y, m, d = rag.data_asof(courses).split("-")
head_l, head_r = st.columns([3, 2], vertical_alignment="bottom")
head_l.title("🎓 강의 추천 챗봇")
head_r.markdown(
    f"<div style='text-align:right;opacity:.75;font-size:.9rem'>📅 {y}년 {int(m)}월 {int(d)}일 자료 기준 · 강의 {len(courses)}개<br>"
    "패스트캠퍼스 공개 강의 정보 기반 · 비공식</div>",
    unsafe_allow_html=True,
)

# ── 적용 중인 조건 (누르면 해제) ──────────────────────────────────────────────
f_now = current_filters()
active = [(key, FILTERS[key][0], f_now[key]) for key in FILTERS if f_now[key] != FILTERS[key][1]]
if active:
    n_ok = len(allowed_ids(f_now))
    with st.container(horizontal=True, vertical_alignment="center"):
        st.caption(f"적용 중인 조건 · 맞는 강의 {n_ok}개", width="content")
        for key, label, value in active:
            text = ", ".join(value) if isinstance(value, list) else value
            st.button(f"{label}: {text}  ✕", key=f"chip_{key}", on_click=clear_filter, args=(key,), type="secondary")
        st.button("모두 지우기", key="chip_all", on_click=clear_all_filters, type="tertiary")


# ── 추천 결과 그리기 ──────────────────────────────────────────────────────────
LEVEL_COLOR = {"누구나": "green"}


def show_card(col, pick):
    with col.container(border=True, height="stretch"):
        st.badge(pick["label"], color="primary" if pick["rank"] == 1 else "gray")
        if pick.get("image"):
            st.image(pick["image"], width="stretch")
        st.markdown(f"**{pick['title']}**")
        with st.container(horizontal=True):
            if pick.get("level"):
                st.badge(pick["level"], color=LEVEL_COLOR.get(pick["level"], "orange"), icon=":material/person:")
            if pick.get("hours"):
                st.badge(pick["hours"], color="blue", icon=":material/schedule:")
            if pick.get("subcategory") or pick.get("category"):
                st.badge(pick.get("subcategory") or pick["category"], color="gray")
        if pick.get("reason"):
            st.write(pick["reason"])
        st.link_button("강의 보기", pick["url"], width="stretch", type="primary" if pick["rank"] == 1 else "secondary")


def show_answer(msg, is_last):
    data = msg["data"]
    st.markdown(f"#### {data['summary']}")
    if data["ask_back"]:
        st.info(data["ask_back"], icon="💬")

    picks = data["picks"]
    if picks:
        cols = st.columns(3)
        for col, pick in zip(cols, picks):
            show_card(col, pick)
    if data["detail"]:
        with st.expander("자세한 설명"):
            st.markdown(data["detail"])

    # 못 찾았을 때: 어떤 조건을 빼면 강의가 나오는지 알려준다
    if not picks and data.get("relax") and is_last:
        st.caption("조건을 하나 빼면 이만큼 있습니다.", width="content")
        with st.container(horizontal=True):
            for key, label, count in data["relax"]:
                st.button(f"{label} 조건 빼기 → {count}개", key=f"relax_{msg['id']}_{key}", on_click=clear_filter, args=(key, True))

    # 이어서 물어보기 (가장 최근 답변에만)
    if is_last and picks:
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption("이어서 좁혀 보기", width="content")
            if "누구나" in LEVELS and ss.f_level != ["누구나"]:
                st.button("더 쉬운 강의만", key=f"easy_{msg['id']}", on_click=narrow, args=("f_level", ["누구나"]))
            if ss.f_time == "전체":
                st.button("10시간 이하만", key=f"short_{msg['id']}", on_click=narrow, args=("f_time", "10시간 이하"))
            st.button("다른 강의 더 보기", key=f"more_{msg['id']}", on_click=more_courses, args=([p["url"] for p in picks],))

    with st.container(horizontal=True, vertical_alignment="center"):
        st.caption("이 추천이 도움이 됐나요?", width="content")
        st.feedback("thumbs", key=f"fb_{msg['id']}", on_change=save_feedback, args=(msg["id"],))

    if operator:
        with st.expander(f"🔎 검색 근거 · 검색된 강의 {len(data['retrieved'])}개 (운영자 보기)"):
            if data.get("filters_text"):
                st.caption("적용 조건: " + data["filters_text"])
            picked = {p["url"] for p in picks}
            for r in data["retrieved"]:
                mark = "✅ " if r["url"] in picked else ""
                st.markdown(f"{mark}**[{r['title']}]({r['url']})** · {r['category']} · 유사도 {r['score']:.2f}")
                st.caption(r["snippet"])
            st.caption("추천 생성: " + ("AI (" + data["model"] + ")" if data["ai"] else "검색 순위 그대로 (AI 미사용)"))


def write_log(msg_id, question, retrieved, data, filters_text):
    LOG.parent.mkdir(exist_ok=True)
    row = {
        "id": msg_id,
        "time": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "filters": filters_text,
        "results": [{"title": r["title"], "url": r["url"], "score": round(r["score"], 3)} for r in retrieved],
        "picks": [{"title": p["title"], "url": p["url"], "label": p["label"]} for p in data["picks"]],
        "answer": data["summary"] + ("\n\n" + data["detail"] if data["detail"] else ""),
        "search_mode": index.mode,
    }
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_question(q):
    """검색(R) -> 추천 생성(A, G) -> 화면에 그릴 자료로 정리"""
    f = current_filters()
    filters_text = " / ".join(
        f"{FILTERS[key][0]}={', '.join(f[key]) if isinstance(f[key], list) else f[key]}" for key in FILTERS if f[key] != FILTERS[key][1]
    )
    retrieved = index.search(q, k=k, allowed=allowed_ids(f, ss.exclude))
    history = [{"role": m["role"], "content": m["content"]} for m in ss.msgs[-6:]]
    rec = rag.recommend(q, retrieved, history=history)

    keep = ["title", "url", "category", "subcategory", "level", "hours", "image"]
    picks = []
    for rank, p in enumerate(rec["picks"], 1):
        r = retrieved[p["n"] - 1]  # 제목·링크·수강 대상은 AI가 쓴 글이 아니라 수집 데이터에서 가져온다
        picks.append({**{key: r.get(key, "") for key in keep}, "label": p["label"], "reason": p["reason"], "rank": rank})

    relax = []
    if not retrieved:  # 조건을 하나씩 빼 보며 몇 개가 나오는지 계산
        for key in FILTERS:
            if f[key] != FILTERS[key][1]:
                ids = allowed_ids({**f, key: FILTERS[key][1]}, ss.exclude)
                count = len(courses) if ids is None else len(ids)
                if count:
                    relax.append((key, FILTERS[key][0], count))
        if ss.exclude and not relax:
            rec["summary"] = "더 보여드릴 강의가 없습니다. 조건을 바꾸거나 새로 질문해 주세요."

    data = {
        **rec, "picks": picks, "relax": relax, "filters_text": filters_text,
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "retrieved": [{"title": r["title"], "url": r["url"], "category": r["category"], "score": r["score"],
                       "snippet": r["snippets"][0][:200] + "…"} for r in retrieved],
    }
    msg_id = uuid.uuid4().hex[:12]
    if save_log:
        try:
            write_log(msg_id, q, retrieved, data, filters_text)
        except Exception as e:  # 기록 실패가 답변을 막지 않게
            st.caption(f"질문 기록 저장 실패: {e}")
    # 다음 질문의 맥락으로 넘길 요약 글
    memo = data["summary"] + "".join(f"\n- {p['title']} ({p['label']})" for p in picks)
    return {"role": "assistant", "content": memo, "data": data, "id": msg_id}


# ── 대화 ──────────────────────────────────────────────────────────────────────
typed = st.chat_input("무엇을 배우고 싶으세요?")
if typed:
    ss.exclude = []
question = typed or ss.pending
ss.pending = None

if not ss.msgs and not question:
    st.markdown("##### 이렇게 물어보세요")
    with st.container(horizontal=True):
        for i, ex in enumerate(EXAMPLES):
            st.button(ex, key=f"ex_{i}", on_click=ask, args=(ex,))

for i, msg in enumerate(ss.msgs):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            show_answer(msg, is_last=(i == len(ss.msgs) - 1 and not question))

if question:
    st.chat_message("user").markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("강의를 찾는 중…"):
            reply = run_question(question)
    ss.msgs += [{"role": "user", "content": question}, reply]
    st.rerun()  # 방금 답변을 버튼·평가와 함께 다시 그린다

st.caption("AI가 생성한 추천이라 틀릴 수 있습니다. 가격과 일정은 강의 페이지에서 확인하세요. "
           + ("질문과 추천 결과는 품질 개선을 위해 저장됩니다. 개인정보는 입력하지 마세요." if save_log else ""))
