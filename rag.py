"""2단계: 검색(Retrieval) + 생성(Generation) 핵심 로직.

R  검색 : 강의 소개글을 조각(chunk)으로 나눠 색인하고, 질문과 가장 비슷한 조각을 찾는다.
          - 키워드 검색(TF-IDF, 글자 n-gram)  + 의미 검색(임베딩, 설치돼 있으면)
A  증강 : 찾은 강의 정보를 프롬프트에 붙인다.
G  생성 : LLM(OpenAI gpt-4o-mini)이 그 정보만 근거로 추천 답변을 쓴다.
"""
import hashlib, json, os, re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DATA = Path(__file__).parent / "data"
# 임베딩 방식: .env 의 EMBEDDING=openai (기본) 또는 EMBEDDING=local
LOCAL_EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # PC에서 실행, 무료
OPENAI_EMB_MODEL = "text-embedding-3-small"  # OpenAI API 호출, 소액 과금
OPENAI_EMB_DIM = 512  # 벡터 크기(기본 1536을 줄여 파일을 작게)
CHUNK, OVERLAP = 700, 100

SYSTEM = """당신은 패스트캠퍼스 강의 추천 도우미입니다.
아래 <강의목록>에 있는 강의만 추천하세요. 목록에 없는 강의나 사실(가격, 기간 등)을 지어내지 마세요.
- 사용자의 목적·수준에 맞는 강의를 1~3개 골라, 각각 왜 맞는지 근거를 2~3문장으로 설명합니다.
- 각 추천에는 강의 제목과 링크를 반드시 붙입니다.
- 맞는 강의가 없으면 솔직히 없다고 말하고, 어떤 정보를 더 알려주면 좋을지 되물어 주세요.
- <강의목록>이 비어 있으면 조건에 맞는 강의가 없다는 뜻입니다. 조건을 넓혀 보라고 안내하세요.
- 가격 정보는 목록에 없습니다. 가격을 물으면 지어내지 말고 링크에서 확인하라고 안내합니다."""


def split(text):
    step = CHUNK - OVERLAP
    return [text[i : i + CHUNK] for i in range(0, max(len(text), 1), step)]


def _norm(x):
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 0 else np.zeros_like(x)


class CourseIndex:
    def __init__(self, use_embeddings=True):
        self.courses = json.loads((DATA / "courses.json").read_text(encoding="utf-8"))
        self.chunks, self.owner = [], []
        for ci, c in enumerate(self.courses):
            head = f"{c['title']} | {c['category']}"
            if c["description"] and c["description"] not in c["text"]:
                head += f" | {c['description']}"
            for piece in split(c["text"]):
                self.chunks.append(f"{head}\n{piece}")
                self.owner.append(ci)
        self.owner = np.array(self.owner)

        self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), sublinear_tf=True)
        self.tfidf_mat = self.tfidf.fit_transform(self.chunks)

        # 임베딩: 원하는 방식부터 시도하고, 안 되면 다음 방식 -> 그래도 안 되면 키워드 검색만
        self.backend, self.emb, self._local, self.embed_stats = None, None, None, None
        if use_embeddings:
            want = os.getenv("EMBEDDING", "openai").strip().lower()
            for backend in ([want] + [b for b in ("openai", "local") if b != want]):
                try:
                    self._load_embeddings(backend)
                    self.backend = backend
                    break
                except Exception as e:
                    print(f"{backend} 임베딩을 사용할 수 없습니다: {e}")
                    self.emb = None

    def _embed(self, texts, backend):
        """글 목록 -> 길이 1로 맞춘 벡터 배열"""
        if backend == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY 가 없습니다")
            from openai import OpenAI

            client, out = OpenAI(), []
            for i in range(0, len(texts), 200):  # 한 번에 200개씩 요청
                res = client.embeddings.create(
                    model=OPENAI_EMB_MODEL, input=texts[i : i + 200], dimensions=OPENAI_EMB_DIM
                )
                out += [d.embedding for d in res.data]
            emb = np.array(out, dtype=np.float32)
        elif backend == "local":
            if self._local is None:
                from fastembed import TextEmbedding

                self._local = TextEmbedding(LOCAL_EMB_MODEL)
            emb = np.array(list(self._local.embed(texts)), dtype=np.float32)
        else:
            raise ValueError(f"알 수 없는 임베딩 방식: {backend}")
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    def _load_embeddings(self, backend):
        """조각마다 지문(해시)을 저장해 두고, 지문이 같은 조각은 예전 벡터를 그대로 쓴다.
        새로 생기거나 내용이 바뀐 조각만 임베딩한다."""
        if backend == "openai" and not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY 가 없습니다")  # 질문 임베딩에도 키가 필요
        model = f"{OPENAI_EMB_MODEL}:{OPENAI_EMB_DIM}" if backend == "openai" else LOCAL_EMB_MODEL
        vec_file, meta_file = DATA / f"embeddings_{backend}.npy", DATA / f"embeddings_{backend}.json"
        hashes = [hashlib.sha256(c.encode("utf-8")).hexdigest() for c in self.chunks]

        old = {}  # 지문 -> 예전 벡터
        if vec_file.exists() and meta_file.exists():
            try:
                meta, vecs = json.loads(meta_file.read_text(encoding="utf-8")), np.load(vec_file)
                if meta.get("model") == model and len(meta["hashes"]) == len(vecs):
                    old = dict(zip(meta["hashes"], vecs))
            except Exception as e:
                print(f"저장된 임베딩을 읽지 못해 새로 계산합니다: {e}")

        todo = [i for i, h in enumerate(hashes) if h not in old]
        new = self._embed([self.chunks[i] for i in todo], backend) if todo else []
        fresh = dict(zip(todo, new))
        self.emb = np.array([fresh[i] if i in fresh else old[h] for i, h in enumerate(hashes)], dtype=np.float32)
        self.embed_stats = {"reused": len(hashes) - len(todo), "computed": len(todo), "removed": len(set(old) - set(hashes))}
        print(f"임베딩({backend}): 재사용 {self.embed_stats['reused']}개 / 새로 계산 {len(todo)}개 / 삭제 {self.embed_stats['removed']}개")

        if todo or len(old) != len(set(hashes)):  # 달라진 게 있을 때만 다시 저장
            np.save(vec_file, self.emb)
            meta_file.write_text(json.dumps({"model": model, "hashes": hashes}), encoding="utf-8")
        if backend == "local" and not todo:
            self._embed(["준비"], backend)  # 질문 임베딩에 쓸 모델을 미리 불러 둠

    @property
    def mode(self):
        if self.emb is None:
            return "키워드 검색"
        name = f"OpenAI {OPENAI_EMB_MODEL}" if self.backend == "openai" else "로컬 모델"
        return f"키워드 + 의미 검색 ({name})"

    def search(self, query, k=5, allowed=None):
        """allowed: 검색 대상으로 허용할 강의 번호 모음 (조건 필터). None 이면 전체."""
        score = _norm((self.tfidf_mat @ self.tfidf.transform([query]).T).toarray().ravel())
        if self.emb is not None:
            try:
                q = self._embed([query], self.backend)[0]
                score = 0.4 * score + 0.6 * _norm(self.emb @ q)
            except Exception as e:  # 질문 임베딩 실패 시 이번 질문은 키워드 검색만
                print(f"질문 임베딩 실패, 키워드 검색만 사용: {e}")
        order = np.argsort(-score)
        if allowed is not None:  # 조건에 맞는 강의의 조각만 남김
            ok = np.isin(self.owner, list(allowed))
            order = [i for i in order if ok[i]]
        best = {}  # 강의별로 가장 점수 높은 조각 2개까지
        for i in order:
            hits = best.setdefault(int(self.owner[i]), [])
            if len(hits) < 2:
                hits.append(int(i))
            if len(best) >= k and all(len(h) == 2 for h in list(best.values())[:k]):
                break
        out = []
        for ci, hits in list(best.items())[:k]:
            c = self.courses[ci]
            out.append({**c, "score": float(score[hits[0]]),
                        "snippets": [self.chunks[i].split("\n", 1)[1] for i in hits]})
        return out


def hours_number(course):
    """'약 12시간' -> 12.0, '약 40분' -> 0.67, 정보가 없으면 None"""
    m = re.search(r"(\d+)\s*(시간|분)", course.get("hours") or "")
    if not m:
        return None
    return float(m.group(1)) if m.group(2) == "시간" else round(int(m.group(1)) / 60, 2)


def data_asof(courses):
    """수집 기준일 'YYYY-MM-DD'. 수집 시각이 없는 예전 데이터면 파일 수정일."""
    stamps = [c["crawled_at"] for c in courses if c.get("crawled_at")]
    if stamps:
        return max(stamps)[:10]
    from datetime import datetime

    return datetime.fromtimestamp((DATA / "courses.json").stat().st_mtime).strftime("%Y-%m-%d")


def build_context(results):
    blocks = []
    for n, r in enumerate(results, 1):
        body = " … ".join(r["snippets"])
        lines = [f"[{n}] 제목: {r['title']}", f"링크: {r['url']}", f"분야: {r['category']} ({r['type']})"]
        if r.get("level") and "수강 대상:" not in body:
            lines.append(f"수강 대상: {r['level']}")
        if r.get("hours") and "수강 시간:" not in body:
            lines.append(f"수강 시간: {r['hours']}")
        if r["description"] and r["description"] not in body:
            lines.append(f"한줄소개: {r['description']}")
        lines.append(f"관련 내용: {body}")
        blocks.append("\n".join(lines))
    return "<강의목록>\n" + "\n\n".join(blocks) + "\n</강의목록>"


def answer(question, results, history=()):
    """OpenAI 모델(기본 gpt-4o-mini)로 추천 답변 생성 (스트리밍). API 키가 없으면 None."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    from openai import OpenAI

    client = OpenAI()
    messages = [
        {"role": "system", "content": SYSTEM},
        *history,
        {"role": "user", "content": f"{build_context(results)}\n\n질문: {question}"},
    ]

    def gen():
        stream = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            max_tokens=1200, messages=messages, stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return gen()


# ── 카드형 추천: AI가 글 대신 정해진 형식(JSON)으로 답한다 ───────────────────────
SYSTEM_CARDS = """당신은 패스트캠퍼스 강의 추천 도우미입니다.
<강의목록>에 있는 강의만 추천하고, 목록에 없는 강의나 사실(가격, 일정 등)은 지어내지 마세요.
반드시 아래 형식의 JSON 하나만 출력합니다.

{
  "summary": "한두 문장 결론. 어떤 강의가 왜 가장 맞는지. 맞는 강의가 없으면 그렇게 말하기",
  "picks": [
    {"n": 강의목록의 번호, "label": "가장 적합", "reason": "이 사람에게 이 강의가 맞는 이유 한두 문장"}
  ],
  "detail": "덧붙일 설명이 있을 때만. 없으면 빈 문자열",
  "ask_back": "질문이 너무 막연해서 직무나 수준을 알아야 할 때만 되묻는 질문 한 문장. 아니면 빈 문자열"
}

규칙
- picks 는 질문에 실제로 맞는 강의만 1~3개. 맞는 순서대로. 억지로 3개를 채우지 않습니다. 맞는 게 없으면 빈 배열.
- label 은 10자 이내로 그 강의의 성격을 나타냅니다. 예: 가장 적합, 짧게 배우기, 더 깊이, 입문용, 실습 중심.
- reason 은 강의목록에 적힌 내용(수강 대상, 수강 시간, 키워드, 소개)을 근거로 씁니다.
- 가격은 목록에 없습니다. 가격을 물으면 summary 에 강의 페이지에서 확인하라고 안내합니다."""

RANK_LABELS = ["가장 가까운 강의", "두 번째", "세 번째"]


def _fallback(results, note):
    """AI 없이 검색 순위만으로 카드를 만든다 (키 없음, 호출 실패 등)."""
    picks = [{"n": n + 1, "label": RANK_LABELS[n], "reason": (r.get("description") or "")[:110]} for n, r in enumerate(results[:3])]
    return {"summary": note, "picks": picks, "detail": "", "ask_back": "", "ai": False}


def recommend(question, results, history=()):
    """검색된 강의 중에서 추천할 강의와 이유를 정한다. 항상 같은 모양의 dict 를 돌려준다."""
    if not results:
        return {"summary": "조건에 맞는 강의가 없습니다.", "picks": [], "detail": "", "ask_back": "", "ai": False}
    if not os.getenv("OPENAI_API_KEY"):
        return _fallback(results, "API 키가 없어 추천 설명 없이 검색 순위대로 보여드립니다.")
    try:
        from openai import OpenAI

        res = OpenAI().chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            max_tokens=900,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_CARDS},
                *history,
                {"role": "user", "content": f"{build_context(results)}\n\n질문: {question}"},
            ],
        )
        raw = json.loads(res.choices[0].message.content)
    except Exception as e:
        print(f"추천 생성 실패, 검색 순위로 대체: {e}")
        return _fallback(results, "추천 설명을 만들지 못해 검색 순위대로 보여드립니다.")

    picks, seen = [], set()
    for p in raw.get("picks") or []:
        try:
            n = int(p.get("n"))
        except Exception:
            continue
        if 1 <= n <= len(results) and n not in seen:  # 목록에 없는 번호는 버림
            seen.add(n)
            picks.append({"n": n, "label": str(p.get("label") or "추천")[:12], "reason": str(p.get("reason") or "").strip()})
    return {
        "summary": str(raw.get("summary") or "").strip() or "추천 결과입니다.",
        "picks": picks[:3],
        "detail": str(raw.get("detail") or "").strip(),
        "ask_back": str(raw.get("ask_back") or "").strip(),
        "ai": True,
    }
