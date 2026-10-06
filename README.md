# 패스트캠퍼스 강의 추천 챗봇 (RAG 시연용)

공개된 강의 소개 페이지를 모아 검색하고, OpenAI 모델(gpt-4o-mini)이 그 내용만 근거로 강의를 추천합니다. 비공식 시연용입니다.

## 실행 방법 (Windows, uv 필요)

- **처음:** `setup.bat` 더블클릭 → 패키지 설치, 수업용 2개 분야 수집, API 키 입력 안내, 챗봇 실행까지 한 번에 진행
- **다음부터:** `run.bat` 더블클릭
- **테스트용 100개 수집:** `crawl_test.bat` 더블클릭 (분야별로 고르게 100개)
- **수업용 2개 분야 수집:** `crawl_class.bat` 더블클릭 (AI/업무생산성, 비즈니스/기획)
- **전체 강의 수집:** `crawl_all.bat` 더블클릭
- 수집 후에는 챗봇을 재시작하세요. 임베딩은 자동으로 다시 계산됩니다.

명령어로 직접 실행하려면:

```
uv sync
uv run python crawl.py --limit 100
uv run streamlit run app.py
```

추천 설명을 생성하려면 `.env` 를 메모장으로 열어 `OPENAI_API_KEY` 를 입력하세요. 모델을 바꾸려면 `OPENAI_MODEL=모델이름` 줄을 추가합니다.

- API 키가 없어도 실행됩니다. 이 경우 검색된 강의 목록만 보여줍니다.
- 임베딩은 기본으로 OpenAI(`text-embedding-3-small`)를 씁니다. `.env` 에 `EMBEDDING=local` 을 넣으면 PC에서 도는 무료 모델(처음에 약 200MB 다운로드)로 바뀝니다.
- 임베딩이 모두 실패하면 키워드 검색만으로 동작합니다.
- 강의를 다시 수집하면 새로 생기거나 내용이 바뀐 조각만 다시 임베딩합니다. 나머지는 저장해 둔 벡터(`data/embeddings_openai.npy`)를 그대로 씁니다. 사이드바에 재사용·새로 계산·삭제 개수가 표시됩니다.

## 파일 구성

| 파일 | 역할 | RAG 단계 |
|---|---|---|
| `crawl.py` | 카테고리 목록 데이터에서 전체 강의 수집 → `data/courses.json` | 데이터 준비 |
| `rag.py` | 조각내기, 색인, 검색, 프롬프트 구성, 답변 생성 | R · A · G |
| `app.py`, `pages/` | 챗봇 화면(한 줄 결론, 추천 카드, 이어서 좁혀 보기, 평가, 운영자 보기), 수집 데이터 화면, 질문 기록 화면 (Streamlit) | 화면 |
| `common.py` | 모든 화면 공통 준비: 키 읽기, 접속 암호 | 화면 |
| `logs/` | 질문·추천 결과 기록(`questions.jsonl`), 도움됨/안 됨 평가(`feedback.jsonl`). 실행하면 생김, GitHub 제외 | 운영 |
| `build_web.py` | 웹 버전용 데이터 묶음 생성 → `web/data/index.json` | 배포 준비 |
| `web/` | Vercel 배포용 웹 버전 (화면 `public/index.html`, 서버 `api/chat.js`) | 배포 |

## 배포

두 가지 버전이 같은 데이터(`data/courses.json`)를 씁니다.

| | Streamlit 버전 | 웹 버전 (`web/`) |
|---|---|---|
| 용도 | PC 실습, 수업 시연 | 공개 링크 |
| 배포처 | Streamlit Community Cloud | Vercel |
| 임베딩 | OpenAI 또는 로컬 | OpenAI만 |

### 공통: GitHub에 올리기 전

1. `.env` 는 `.gitignore` 로 제외돼 있습니다. 키가 올라가지 않았는지 꼭 확인하세요.
2. 저장소는 **비공개(Private)** 를 권합니다. `data/` 와 `web/data/` 에 강의 소개글이 들어 있습니다.
3. OpenAI 계정에 월 사용 한도를 설정해 두세요. 공개 링크는 누구나 질문할 수 있습니다.

### Streamlit Community Cloud

1. share.streamlit.io 에서 **Create app** → 저장소 `mjcho7/fastchatbot`, 브랜치 `main`, 실행 파일 `app.py`
2. **Advanced settings** 에서 Python 3.12 선택, Secrets 에 아래 내용 입력

```
OPENAI_API_KEY = "키"
APP_PASSWORD = "접속 암호"
```

- `APP_PASSWORD` 는 선택입니다. 넣으면 암호를 아는 사람만 챗봇을 쓸 수 있습니다.
- 패키지는 `uv.lock` 기준으로 설치됩니다.

### Vercel

1. `build_web.bat` 을 실행해 `web/data/index.json` 을 만들고 GitHub 에 올립니다. (강의를 다시 수집할 때마다 반복)
2. Vercel 에서 저장소를 가져올 때 **Root Directory 를 `web`** 으로 지정합니다.
3. 환경변수에 `OPENAI_API_KEY` 를 넣습니다.
4. 선택: `ACCESS_CODE` 를 넣으면 그 코드를 아는 사람만 질문할 수 있습니다. 수업용 링크라면 설정을 권합니다.

## 수집 관련 유의사항

- 사이트가 카테고리 화면에 쓰는 목록 데이터를 카테고리별로 1번씩(약 10번) 요청합니다. 강의 상세 페이지는 열지 않습니다.
- 공식 공개 API가 아니어서 예고 없이 바뀌거나 막힐 수 있습니다.
- 이용약관은 회사 정보를 영리 목적으로 복제·배포하는 것을 금지합니다. 수집 데이터는 교육·시연에만 쓰고 외부에 배포하지 마세요.
- 가격 정보는 수집하지 않습니다. 챗봇은 가격을 링크에서 확인하도록 안내합니다.
