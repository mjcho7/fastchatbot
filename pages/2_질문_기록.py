"""저장된 질문 기록을 보는 화면. 사람들이 무엇을 묻는지, 어디서 못 찾는지 확인하는 용도."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import common  # noqa: E402

LOG = ROOT / "logs" / "questions.jsonl"

st.set_page_config(page_title="질문 기록", page_icon="📝", layout="wide")
common.setup()
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
for col, empty in (("id", ""), ("filters", ""), ("picks", None)):
    if col not in df:
        df[col] = empty

# 평가(도움됨/안 됨): 같은 답변을 여러 번 누르면 마지막 것만 사용
FEEDBACK = LOG.parent / "feedback.jsonl"
votes = {}
if FEEDBACK.exists():
    for line in FEEDBACK.read_text(encoding="utf-8").splitlines():
        try:
            v = json.loads(line)
            votes[v["id"]] = "👍" if v["helpful"] else "👎"
        except Exception:
            pass
df["평가"] = df["id"].map(lambda i: votes.get(i, ""))
df["추천한 강의"] = df["picks"].map(lambda p: ", ".join(x["title"] for x in p) if isinstance(p, list) else "")
df["검색된 강의 수"] = df["results"].map(len)
df["1위 강의"] = df["results"].map(lambda r: r[0]["title"] if r else "")
df["1위 유사도"] = df["results"].map(lambda r: r[0]["score"] if r else None)
df["조건"] = df["filters"].map(lambda f: f if isinstance(f, str) else ", ".join(f"{k}={v}" for k, v in (f or {}).items() if v and v != "전체"))

c1, c2, c3, c4 = st.columns(4)
c1.metric("질문 수", f"{len(df)}건")
c2.metric("강의를 못 찾은 질문", f"{int((df['검색된 강의 수'] == 0).sum())}건")
c3.metric("도움 안 됨 👎", f"{int((df['평가'] == '👎').sum())}건")
c4.metric("도움됨 👍", f"{int((df['평가'] == '👍').sum())}건")
st.caption(f"기간: {df['time'].min()[:10]} ~ {df['time'].max()[:10]}")
st.caption(f"파일: {LOG}")

f1, f2 = st.columns([2, 1])
word = f1.text_input("질문에서 찾기")
only = f2.selectbox("보기", ["전체", "도움 안 됨 👎 만", "강의를 못 찾은 질문만"])
view = df[df["question"].str.contains(word, case=False, regex=False)] if word else df
if only.startswith("도움 안 됨"):
    view = view[view["평가"] == "👎"]
elif only.startswith("강의를 못"):
    view = view[view["검색된 강의 수"] == 0]
if view.empty:
    st.info("조건에 맞는 질문이 없습니다.")
    st.stop()
view = view.iloc[::-1]  # 최근 질문이 위로

st.dataframe(
    view[["time", "question", "평가", "조건", "추천한 강의", "검색된 강의 수", "1위 유사도"]],
    column_config={"time": "시각", "question": st.column_config.TextColumn("질문", width="large")},
    hide_index=True, width="stretch",
)
st.download_button(
    "CSV로 내려받기 (엑셀용)",
    view[["time", "question", "평가", "조건", "추천한 강의", "검색된 강의 수", "1위 강의", "1위 유사도", "answer"]].to_csv(index=False).encode("utf-8-sig"),
    "questions.csv", "text/csv",
)

st.subheader("질문 상세")
pick = st.selectbox("질문 선택", view.index, format_func=lambda i: f"{df.loc[i, 'time']}  {df.loc[i, 'question'][:50]}")
row = df.loc[pick]
st.markdown(f"**질문:** {row['question']}  {row['평가']}")
if row["조건"]:
    st.caption("조건: " + row["조건"])
st.markdown("**검색된 강의**")
for r in row["results"]:
    st.markdown(f"- [{r['title']}]({r['url']}) · 유사도 {r['score']}")
st.markdown("**답변**")
st.markdown(row["answer"])
