// Vercel 서버 함수: 질문 임베딩 -> 강의 검색 -> gpt-4o-mini 답변 (스트리밍)
// 필요한 환경변수: OPENAI_API_KEY  (선택: ACCESS_CODE, OPENAI_MODEL)
const data = require("../data/index.json");

const CHAT_MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";
const MAX_QUESTION = 500; // 질문 글자 수 제한
const TOP_K = 5;

// base64 로 저장된 벡터를 한 번만 풀어 둔다
let vectors = null;
function getVectors() {
  if (!vectors) {
    const buf = Buffer.from(data.vectors, "base64");
    vectors = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  }
  return vectors;
}

async function openai(path, body) {
  const r = await fetch("https://api.openai.com/v1/" + path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: "Bearer " + process.env.OPENAI_API_KEY },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error("OpenAI " + r.status + ": " + (await r.text()).slice(0, 200));
  return r;
}

// R: 질문과 가장 비슷한 강의 찾기 (의미 유사도 + 질문 단어가 그대로 들어 있으면 가산점)
async function search(query, k) {
  const res = await (await openai("embeddings", { model: data.model, input: query, dimensions: data.dim })).json();
  const q = res.data[0].embedding;
  const v = getVectors();
  const words = query.toLowerCase().split(/[\s,.?!]+/).filter((w) => w.length >= 2);
  const best = new Map(); // 강의 번호 -> {score, chunk}
  data.chunks.forEach((ch, i) => {
    let dot = 0;
    for (let d = 0, o = i * data.dim; d < data.dim; d++) dot += v[o + d] * q[d];
    const text = (data.courses[ch.c].title + " " + ch.t).toLowerCase();
    const hits = words.filter((w) => text.includes(w)).length;
    const score = dot + (words.length ? (0.1 * hits) / words.length : 0);
    const cur = best.get(ch.c);
    if (!cur || score > cur.score) best.set(ch.c, { score, text: ch.t });
  });
  return [...best.entries()]
    .sort((a, b) => b[1].score - a[1].score)
    .slice(0, k)
    .map(([c, m]) => ({ ...data.courses[c], score: Math.round(m.score * 100) / 100, snippet: m.text }));
}

// A: 찾은 강의를 프롬프트에 붙이기
function buildContext(results) {
  const blocks = results.map(
    (r, n) => `[${n + 1}] 제목: ${r.title}\n링크: ${r.url}\n분야: ${r.category} (${r.type})\n관련 내용: ${r.snippet}`
  );
  return "<강의목록>\n" + blocks.join("\n\n") + "\n</강의목록>";
}

module.exports = async (req, res) => {
  if (req.method !== "POST") return res.status(405).json({ error: "POST 만 지원합니다" });
  if (!process.env.OPENAI_API_KEY) return res.status(500).json({ error: "서버에 OPENAI_API_KEY 가 설정되지 않았습니다" });

  const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : req.body || {};
  if (process.env.ACCESS_CODE && body.code !== process.env.ACCESS_CODE)
    return res.status(401).json({ error: "접속 코드가 필요합니다" });

  const question = String(body.question || "").trim().slice(0, MAX_QUESTION);
  if (!question) return res.status(400).json({ error: "질문을 입력하세요" });
  const history = (Array.isArray(body.history) ? body.history : [])
    .filter((m) => m && (m.role === "user" || m.role === "assistant") && typeof m.content === "string")
    .slice(-6)
    .map((m) => ({ role: m.role, content: m.content.slice(0, 2000) }));

  try {
    // 후속 질문("그중 더 쉬운 건?")도 검색되도록 직전 질문을 함께 검색어로 사용
    const prev = history.filter((m) => m.role === "user").slice(-1).map((m) => m.content.slice(0, MAX_QUESTION));
    const results = await search([...prev, question].join(" "), TOP_K);

    const upstream = await openai("chat/completions", {
      model: CHAT_MODEL,
      max_tokens: 1200,
      stream: true,
      messages: [
        { role: "system", content: data.system },
        ...history,
        { role: "user", content: buildContext(results) + "\n\n질문: " + question },
      ],
    });

    // 응답 형식: 첫 줄 = 검색된 강의(JSON), 그 뒤 = 답변 글자가 도착하는 대로
    res.writeHead(200, { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store", "X-Accel-Buffering": "no" });
    res.write(JSON.stringify({ sources: results.map(({ snippet, ...r }) => r) }) + "\n");

    const decoder = new TextDecoder();
    let buf = "";
    for await (const part of upstream.body) {
      buf += decoder.decode(part, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ") || line === "data: [DONE]") continue;
        try {
          const delta = JSON.parse(line.slice(6)).choices?.[0]?.delta?.content;
          if (delta) res.write(delta);
        } catch {}
      }
    }
    res.end();
  } catch (e) {
    console.error(e);
    if (res.headersSent) res.end("\n\n(답변 생성 중 오류가 발생했습니다)");
    else res.status(502).json({ error: "답변을 만들지 못했습니다. 잠시 후 다시 시도해 주세요." });
  }
};
