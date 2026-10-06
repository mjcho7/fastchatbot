"""모든 화면이 맨 위에서 부르는 공통 준비: 배포 환경의 비밀 값 읽기 + (선택) 접속 암호."""
import hmac
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).parent
KEYS = ("OPENAI_API_KEY", "OPENAI_MODEL", "EMBEDDING", "APP_PASSWORD")


def setup():
    # PC에서는 .env, Streamlit Community Cloud 에서는 Secrets 에 넣은 값을 환경변수로 맞춘다
    load_dotenv(ROOT / ".env")
    try:
        for k in KEYS:
            if k in st.secrets and not os.getenv(k):
                os.environ[k] = str(st.secrets[k])
    except Exception:
        pass  # secrets 파일이 없는 PC 환경

    # APP_PASSWORD 가 설정돼 있으면 암호를 아는 사람만 사용 (공개 링크의 요금 보호)
    password = os.getenv("APP_PASSWORD")
    if not password or st.session_state.get("authed"):
        return
    st.title("🔒 접속 암호")
    entered = st.text_input("암호를 입력하세요", type="password")
    if entered:
        if hmac.compare_digest(entered, password):
            st.session_state["authed"] = True
            st.rerun()
        st.error("암호가 맞지 않습니다.")
    st.stop()
