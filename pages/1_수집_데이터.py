"""수집한 강의 데이터를 살펴보는 화면 (챗봇 왼쪽 메뉴에서 선택)."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import common  # noqa: E402
import rag  # noqa: E402

st.set_page_config(page_title="수집 데이터", page_icon="📚", layout="wide")
common.setup()
st.title("📚 수집 데이터")

path = ROOT / "data" / "courses.json"
if not path.exists():
    st.warning("수집된 데이터가 없습니다. 먼저 `crawl_all.bat` 또는 `uv run python crawl.py` 를 실행하세요.")
    st.stop()

courses = json.loads(path.read_text(encoding="utf-8"))
df = pd.DataFrame(courses)
df["본문 길이"] = df["text"].str.len()
df["청크 수"] = df["text"].map(lambda t: len(rag.split(t)))

# ── 요약 ──────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("강의 수", f"{len(df)}개")
c2.metric("분야 수", f"{df['category'].nunique()}개")
c3.metric("평균 본문 길이", f"{int(df['본문 길이'].mean()):,}자")
c4.metric("전체 청크 수", f"{int(df['청크 수'].sum()):,}개")
st.caption(f"파일: {path} · 청크 기준: {rag.CHUNK}자 / {rag.OVERLAP}자 겹침")

# ── 필터 ──────────────────────────────────────────
f1, f2 = st.columns([1, 2])
cats = f1.multiselect("분야", sorted(df["category"].unique()))
word = f2.text_input("제목·본문에서 찾기", placeholder="예: 엑셀, 보고서, Claude")
view = df
if cats:
    view = view[view["category"].isin(cats)]
if word:
    hit = view["title"].str.contains(word, case=False, regex=False) | view["text"].str.contains(word, case=False, regex=False)
    view = view[hit]

tab_list, tab_detail, tab_stats = st.tabs([f"목록 ({len(view)})", "강의 상세 · 청크", "분야별 통계"])

with tab_list:
    st.dataframe(
        view[[c for c in ["title", "category", "subcategory", "type", "level", "hours", "본문 길이", "청크 수", "description", "url"] if c in view.columns]],
        column_config={
            "title": st.column_config.TextColumn("제목", width="large"),
            "category": "분야",
            "subcategory": "세부 분류",
            "type": "형태",
            "level": "수강 대상",
            "hours": "수강 시간",
            "description": st.column_config.TextColumn("한줄소개", width="large"),
            "url": st.column_config.LinkColumn("링크", display_text="열기"),
        },
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "목록을 CSV로 내려받기 (엑셀용)",
        view.drop(columns=["text"]).to_csv(index=False).encode("utf-8-sig"),
        "courses.csv", "text/csv",
    )

with tab_detail:
    if view.empty:
        st.info("조건에 맞는 강의가 없습니다.")
    else:
        title = st.selectbox("강의 선택", view["title"].tolist())
        c = view[view["title"] == title].iloc[0]
        left, right = st.columns([1, 3])
        if c["image"]:
            left.image(c["image"])
        right.markdown(f"### {c['title']}\n{c['category']} · {c['type']} · [강의 페이지 열기]({c['url']})")
        right.write(c["description"])

        chunks = rag.split(c["text"])
        st.markdown(f"**본문 {len(c['text']):,}자 → 청크 {len(chunks)}개**")
        mode = st.radio("보기", ["청크로 나눠 보기", "본문 전체 보기"], horizontal=True, label_visibility="collapsed")
        if mode == "본문 전체 보기":
            st.text_area("본문", c["text"], height=400, label_visibility="collapsed")
        else:
            st.caption("각 청크 앞에는 '제목 | 분야 | 한줄소개'가 붙어서 임베딩됩니다. 회색 부분은 앞 청크와 겹치는 구간입니다.")
            for i, ch in enumerate(chunks):
                with st.expander(f"청크 {i + 1} · {len(ch)}자 · {ch[:40]}…", expanded=i == 0):
                    if word and word.lower() in ch.lower():
                        st.success(f"'{word}' 포함")
                    ov = rag.OVERLAP if i > 0 else 0
                    st.markdown(f":gray[{ch[:ov]}]{ch[ov:]}" if ov else ch)

with tab_stats:
    g = view.groupby("category").agg(강의수=("title", "count"), 평균본문길이=("본문 길이", "mean"), 청크수=("청크 수", "sum"))
    g["평균본문길이"] = g["평균본문길이"].round().astype(int)
    st.bar_chart(g["강의수"])
    st.dataframe(g, width="stretch")
