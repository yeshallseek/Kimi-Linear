import { createServer } from "node:http";
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { randomUUID } from "node:crypto";

const PORT = Number(process.env.KIMI_UI_API_PORT || 5174);
const HOST = process.env.KIMI_UI_API_HOST || "0.0.0.0";
const LLAMA_URL = process.env.KIMI_LLAMA_URL || "http://127.0.0.1:8081";
const MODEL_ID = "ymcki/Kimi-Linear-48B-A3B-Instruct-GGUF:MXFP4_MOE";
const repoRoot = resolve(dirname(new URL(import.meta.url).pathname), "..");
const runsPath = resolve(repoRoot, "experiments/runs.jsonl");
const chatArtifactDir = resolve(repoRoot, "artifacts/chat");

function json(res, status, value) {
  const body = JSON.stringify(value);
  res.writeHead(status, {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "content-type",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Content-Type": "application/json",
  });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolveBody, rejectBody) => {
    let data = "";
    req.on("data", (chunk) => {
      data += chunk;
      if (data.length > 2_000_000) {
        rejectBody(new Error("Request body too large"));
        req.destroy();
      }
    });
    req.on("end", () => {
      try {
        resolveBody(data ? JSON.parse(data) : {});
      } catch (error) {
        rejectBody(error);
      }
    });
  });
}

function appendRun(record) {
  appendFileSync(runsPath, `${JSON.stringify(record)}\n`);
}

function recentChatRuns(limit = 20) {
  if (!existsSync(runsPath)) return [];
  return readFileSync(runsPath, "utf8")
    .trim()
    .split("\n")
    .filter(Boolean)
    .slice(-400)
    .map((line) => {
      try {
        return JSON.parse(line);
      } catch {
        return null;
      }
    })
    .filter((record) => record?.type === "kimi_gguf_chat")
    .slice(-limit)
    .reverse();
}

async function llamaStatus() {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 1500);
  try {
    const response = await fetch(`${LLAMA_URL}/v1/models`, { signal: controller.signal });
    const payload = await response.json().catch(() => ({}));
    return { ok: response.ok, status: response.status, payload };
  } catch (error) {
    return { ok: false, error: error.message };
  } finally {
    clearTimeout(timeout);
  }
}

async function chatCompletion(payload) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), Number(payload.timeoutMs || 180000));
  try {
    const response = await fetch(`${LLAMA_URL}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: MODEL_ID,
        messages: payload.messages,
        temperature: payload.temperature,
        top_p: payload.topP,
        max_tokens: payload.maxTokens,
        stream: false,
      }),
      signal: controller.signal,
    });
    const data = await response.json().catch(async () => ({ error: await response.text() }));
    if (!response.ok) {
      throw new Error(data?.error?.message || data?.error || `llama-server returned ${response.status}`);
    }
    return data;
  } finally {
    clearTimeout(timeout);
  }
}

const server = createServer(async (req, res) => {
  if (req.method === "OPTIONS") {
    json(res, 204, {});
    return;
  }

  if (req.method === "GET" && req.url === "/api/status") {
    const status = await llamaStatus();
    json(res, 200, { backend: true, llamaUrl: LLAMA_URL, model: MODEL_ID, llama: status });
    return;
  }

  if (req.method === "GET" && req.url === "/api/history") {
    json(res, 200, { runs: recentChatRuns() });
    return;
  }

  if (req.method === "POST" && req.url === "/api/chat") {
    const runId = `kimi-gguf-${new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14)}-${randomUUID().slice(0, 8)}`;
    const started = Date.now();
    let requestBody = {};

    try {
      requestBody = await readBody(req);
      const messages = Array.isArray(requestBody.messages) ? requestBody.messages : [];
      if (!messages.some((message) => message.role === "user" && String(message.content || "").trim())) {
        throw new Error("Missing user prompt");
      }

      const parameters = {
        temperature: Number(requestBody.temperature ?? 0.7),
        topP: Number(requestBody.topP ?? 0.9),
        maxTokens: Number(requestBody.maxTokens ?? 256),
      };

      const completion = await chatCompletion({ messages, ...parameters });
      const output = completion?.choices?.[0]?.message?.content || "";
      mkdirSync(chatArtifactDir, { recursive: true });
      const artifact = resolve(chatArtifactDir, `${runId}.json`);
      writeFileSync(
        artifact,
        JSON.stringify({ runId, request: { messages, parameters }, response: completion }, null, 2),
      );

      const record = {
        timestamp: new Date().toISOString(),
        type: "kimi_gguf_chat",
        status: "ok",
        run_id: runId,
        model: MODEL_ID,
        artifact,
        inputs: { messages },
        parameters,
        results: {
          latency_ms: Date.now() - started,
          output_preview: output.slice(0, 500),
          usage: completion.usage || null,
        },
      };
      appendRun(record);
      json(res, 200, { runId, output, usage: completion.usage || null, latencyMs: record.results.latency_ms });
    } catch (error) {
      const record = {
        timestamp: new Date().toISOString(),
        type: "kimi_gguf_chat",
        status: "error",
        run_id: runId,
        model: MODEL_ID,
        inputs: requestBody,
        results: { latency_ms: Date.now() - started },
        error: error.message,
      };
      appendRun(record);
      json(res, 500, { error: error.message, runId });
    }
    return;
  }

  json(res, 404, { error: "Not found" });
});

server.listen(PORT, HOST, () => {
  console.log(`Kimi UI API listening on http://${HOST}:${PORT}`);
  console.log(`Proxying llama-server at ${LLAMA_URL}`);
});
