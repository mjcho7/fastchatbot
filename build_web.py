"""Vercel 배포용 데이터 만들기:  data/courses.json -> web/data/index.json

강의 조각을 OpenAI 로 임베딩해서(이미 계산돼 있으면 그대로 사용) 웹 버전이 읽는 파일 하나로 묶습니다.
강의를 다시 수집했으면 이 스크립트도 다시 실행한 뒤 GitHub 에 올리세요.

사용법:  uv run python build_web.py   (또는 build_web.bat 더블클릭)
"""
import base64, json, os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
os.environ["EMBEDDING"] = "openai"  # 웹 버전은 서버에 모델을 둘 수 없어 OpenAI 임베딩만 사용
import rag  # noqa: E402

ix = rag.CourseIndex()
if ix.backend != "openai":
    raise SystemExit("OpenAI 임베딩에 실패했습니다. .env 의 OPENAI_API_KEY 를 확인하세요.")

KEEP = ["title", "url", "category", "type", "level", "hours", "description"]
out = {
    "model": rag.OPENAI_EMB_MODEL,
    "dim": rag.OPENAI_EMB_DIM,
    "system": rag.SYSTEM,
    "courses": [{k: c.get(k, "") for k in KEEP} for c in ix.courses],
    "chunks": [{"c": int(o), "t": t.split("\n", 1)[1]} for o, t in zip(ix.owner, ix.chunks)],
    # 벡터는 float32 를 base64 로 묶어 저장 (JSON 숫자 배열보다 3배쯤 작음)
    "vectors": base64.b64encode(ix.emb.astype("<f4").tobytes()).decode("ascii"),
}
dst = ROOT / "web" / "data" / "index.json"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print(f"완료: 강의 {len(out['courses'])}개 / 조각 {len(out['chunks'])}개 -> {dst} ({dst.stat().st_size / 1e6:.1f}MB)")
