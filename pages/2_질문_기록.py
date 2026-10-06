"""저장된 질문 기록을 보는 화면. 사람들이 무엇을 묻는지, 어디서 못 찾는지 확인하는 용도."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

LOG = Path(__file__).resolve().parent.parent / "logs" / "questions.jsonl"

st.set_page_config(page_title="질문 기록", page_icon="📝", layout="wide")
st.title("📝 질문 기록")

if not LOG.exists():
    st.info("아직 저장된 질문이 없습니다. 챗봇에서 질문하면 여기에 쌓입니다.")
    st.stop()

rows = []
for line in LOG.read_text(encoding="utf-8").splitlines():
    try:
        rows.append(json.loads(line))
    except Exception:
        pass  # 깨진 줄은 건너뜀
if not rows:
    st.info("아직 저장된 질문이 없습니다.")
    st.stop()

df = pd.DataFrame(rows)
df["검색된 강의 수"] = df["results"].map(len)
df["1위 강의"] = df["results"].map(lambda r: r[0]["title"] if r else "")
df["1위 유사도"] = df["results"].map(lambda r: r[0]["score"] if r else None)
df["조건"] = df["filters"].map(lambda f: ", ".join(f"{k}={v}" for k, v in (f or {}).items() if v and v != "전체"))

c1, c2, c3 = st.columns(3)
c1.metric("질문 수", f"{len(df)}건")
c2.metric("강의를 못 찾은 질문", f"{int((df['검색된 강의 수'] == 0).sum())}건")
c3.metric("기간", f"{df['time'].min()[:10]} ~ {df['time'].max()[:10]}")
st.caption(f"파일: {LOG}")

word = st.text_input("질문에서 찾기")
view = df[df["question"].str.contains(word, case=False, regex=False)] if word else df
view = view.iloc[::-1]  # 최근 질문이 위로

st.dataframe(
    view[["time", "question", "조건", "검색된 강의 수", "1위 강의", "1위 유사도"]],
    column_config={"time": "시각", "question": st.column_config.TextColumn("질문", width="large")},
    hide_index=True, width="stretch",
)
st.download_button(
    "CSV로 내려받기 (엑셀용)",
    view[["time", "question", "조건", "검색된 강의 수", "1위 강의", "1위 유사도", "answer"]].to_csv(index=False).encode("utf-8-sig"),
    "questions.csv", "text/csv",
)

st.subheader("질문 상세")
pick = st.selectbox("질문 선택", view.index, format_func=lambda i: f"{df.loc[i, 'time']}  {df.loc[i, 'question'][:50]}")
row = df.loc[pick]
st.markdown(f"**질문:** {row['question']}")
if row["조건"]:
    st.caption("조건: " + row["조건"])
st.markdown("**검색된 강의**")
for r in row["results"]:
    st.markdown(f"- [{r['title']}]({r['url']}) · 유사도 {r['score']}")
st.markdown("**답변**")
st.markdown(row["answer"])
