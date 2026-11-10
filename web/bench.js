const params = new URLSearchParams(location.search);
const WEIGHTS = params.get("weights") === "f32" ? "f32" : "int8";
const BACKEND = params.get("backend") === "webgpu" ? "webgpu" : "cpu";
const PIPELINE = params.get("pipeline") ?? "";
const requestedPkg = params.get("pkg");
const PKG = ["pkg", "pkg-nosimd", "pkg-threads"].includes(requestedPkg)
  ? requestedPkg
  : self.crossOriginIsolated
    ? "pkg-threads"
    : "pkg";
const THREADS = params.get("threads") ?? "";
const BROWSER = {
  name: params.get("browser") ?? "unknown",
  version: params.get("browser_version") ?? browserVersion(),
};
const WARMUP = counterParam("warmup", 1);
const RUNS = counterParam("runs", 5);
const ONLY = params.get("only");

const worker = new Worker(
  new URL(
    `./worker.js?backend=${encodeURIComponent(BACKEND)}&pkg=${encodeURIComponent(PKG)}&threads=${encodeURIComponent(THREADS)}&pipeline=${encodeURIComponent(PIPELINE)}`,
    import.meta.url,
  ),
  { type: "module" },
);

const statusLine = document.getElementById("status");
const rows = document.getElementById("rows");
const summary = document.getElementById("summary");
const downloadButton = document.getElementById("download");

const pending = new Map();
let nextId = 1;
let resolveReady;
let rejectReady;
const ready = new Promise((resolve, reject) => {
  resolveReady = resolve;
  rejectReady = reject;
});

worker.onmessage = (event) => {
  const message = event.data;
  if (message.type === "progress") {
    showProgress(message);
    return;
  }
  if (message.type === "ready") {
    resolveReady(message);
    return;
  }
  if (message.type === "error" && message.id === undefined) {
    rejectReady(new Error(message.message));
    return;
  }

  const request = pending.get(message.id);
  if (!request) {
    return;
  }
  if (message.type === "token") {
    request.onToken?.(message);
    return;
  }
  pending.delete(message.id);
  if (message.type === "error") {
    request.reject(new Error(message.message));
  } else {
    request.resolve(message);
  }
};

worker.onerror = (event) => {
  fail(event.message);
};

function request(message, onToken) {
  const id = nextId;
  nextId += 1;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject, onToken });
    worker.postMessage({ ...message, id });
  });
}

function showProgress(message) {
  const filePart =
    message.total > 0 ? Math.min(1, message.loaded / message.total) : 0;
  const overall = Math.round(((message.index - 1 + filePart) / message.count) * 100);
  const source = message.cached
    ? "from browser storage"
    : `${formatBytes(message.loaded)} of ${formatBytes(message.total || message.loaded)}`;
  statusLine.textContent = `Loading ${message.file}: ${overall}% (${source})`;
}

function formatBytes(bytes) {
  if (bytes >= 1e9) {
    return `${(bytes / 1e9).toFixed(2)} GB`;
  }
  if (bytes >= 1e6) {
    return `${(bytes / 1e6).toFixed(1)} MB`;
  }
  if (bytes >= 1e3) {
    return `${(bytes / 1e3).toFixed(1)} kB`;
  }
  return `${bytes} B`;
}

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${path}: HTTP ${response.status}`);
  }
  return response.json();
}

async function fetchF32(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${path}: HTTP ${response.status}`);
  }
  return new Float32Array(await response.arrayBuffer());
}

function maxAbsDiff(left, right) {
  const count = Math.min(left.length, right.length);
  let max = 0;
  for (let index = 0; index < count; index += 1) {
    const difference = Math.abs(left[index] - right[index]);
    if (difference > max) {
      max = difference;
    }
  }
  return max;
}

function browserVersion() {
  const match = navigator.userAgent.match(
    /(?:Firefox|Chrome|Version)\/(\d+(?:\.\d+)*)/,
  );
  return match ? match[1] : "unknown";
}

function counterParam(name, fallback) {
  const value = Number.parseInt(params.get(name) ?? "", 10);
  return Number.isFinite(value) && value >= 0 ? value : fallback;
}

function compareTokens(actual, expected) {
  let matched = 0;
  let firstMismatch = null;
  for (let index = 0; index < expected.length; index += 1) {
    if (index < actual.length && actual[index] === expected[index]) {
      matched += 1;
    } else if (firstMismatch === null) {
      firstMismatch = index;
    }
  }
  return { matched, firstMismatch };
}

function runMetrics(nPrompt, response) {
  const timings = response.timings;
  const count = response.tokens.length;
  const ttft_ms = timings.length > 0 ? timings[0] : 0;
  const decodeSeconds =
    timings.length > 1 ? (timings[timings.length - 1] - timings[0]) / 1000 : 0;
  return {
    ttft_ms,
    prefill_tok_s: ttft_ms > 0 ? nPrompt / (ttft_ms / 1000) : 0,
    decode_tok_s:
      count > 1 && decodeSeconds > 0 ? (count - 1) / decodeSeconds : 0,
    total_ms: timings.length > 0 ? timings[timings.length - 1] : 0,
    matches: true,
  };
}

function median(values) {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) {
    return sorted[middle];
  }
  return (sorted[middle - 1] + sorted[middle]) / 2;
}

function addRow(prompt) {
  const row = document.createElement("tr");
  const cells = [
    prompt.id,
    String(prompt.n_prompt),
    `${prompt.matched_tokens} / ${prompt.total_tokens}`,
    prompt.first_mismatch === null ? "\u2013" : String(prompt.first_mismatch),
    prompt.logit_max_abs_diff.toExponential(1),
    `${prompt.medians.ttft_ms.toFixed(0)} ms`,
    prompt.medians.prefill_tok_s.toFixed(1),
    prompt.medians.decode_tok_s.toFixed(2),
  ];
  for (const [index, text] of cells.entries()) {
    const cell = document.createElement("td");
    cell.textContent = text;
    if (index === 2 && prompt.first_mismatch !== null) {
      cell.classList.add("warn");
    }
    row.append(cell);
  }
  rows.append(row);
}

async function generate(ids, maxNewTokens) {
  return request({ type: "generate", ids, maxNewTokens, stream: false });
}

async function main() {
  statusLine.textContent = "Loading the engine and weights.";
  worker.postMessage({ type: "load", weights: WEIGHTS });
  const [prompts, expected, readyMessage] = await Promise.all([
    fetchJson("/bench/prompts.json"),
    fetchJson("/reference/expected.json"),
    ready,
  ]);

  const startedAt = new Date();
  const target =
    readyMessage.backend === "webgpu"
      ? "webgpu"
      : readyMessage.threads > 1
        ? "wasm-threads"
        : readyMessage.simd
          ? "wasm-simd"
          : "wasm";
  const maxNewTokens = prompts.max_new_tokens;
  const promptSessions = [];
  let matchesReference = true;
  let nearTieMismatch = false;

  const selected = ONLY
    ? prompts.prompts.filter((prompt) => ONLY.split(",").includes(prompt.id))
    : prompts.prompts;
  if (selected.length === 0) {
    throw new Error(`no prompt matches --only ${ONLY}`);
  }

  for (const prompt of selected) {
    const expectedPrompt = expected.prompts.find(
      (candidate) => candidate.id === prompt.id,
    );
    if (!expectedPrompt) {
      throw new Error(`${prompt.id}: missing from reference/expected.json`);
    }
    const ids = expectedPrompt.prompt_ids;
    statusLine.textContent = `${prompt.id}: checking ${ids.length} prompt tokens.`;

    const logits = await request({ type: "logits", ids });
    const expectedLogits = await fetchF32(
      `/${expectedPrompt.first_logits.replace(/^\/+/, "")}`,
    );
    const logitDiff = maxAbsDiff(logits.values, expectedLogits);
    const logitsOk = logitDiff <= 0.01;

    const check = await generate(ids, maxNewTokens);
    const { matched, firstMismatch } = compareTokens(
      check.tokens,
      expectedPrompt.generated_ids,
    );
    const tokensMatch = matched === expectedPrompt.generated_ids.length;
    if (!logitsOk) {
      matchesReference = false;
    }
    if (!tokensMatch) {
      if (expectedPrompt.near_tie) {
        nearTieMismatch = true;
      } else {
        matchesReference = false;
      }
    }

    for (let index = 0; index < WARMUP; index += 1) {
      statusLine.textContent = `${prompt.id}: warmup ${index + 1} of ${WARMUP}.`;
      await generate(ids, maxNewTokens);
    }

    const runs = [];
    for (let index = 0; index < RUNS; index += 1) {
      statusLine.textContent = `${prompt.id}: timed run ${index + 1} of ${RUNS}.`;
      const response = await generate(ids, maxNewTokens);
      const run = runMetrics(ids.length, response);
      run.matches = compareTokens(
        response.tokens,
        expectedPrompt.generated_ids,
      ).firstMismatch === null;
      runs.push(run);
    }

    const medians = {
      decode_tok_s: median(runs.map((run) => run.decode_tok_s)),
      prefill_tok_s: median(runs.map((run) => run.prefill_tok_s)),
      ttft_ms: median(runs.map((run) => run.ttft_ms)),
    };
    const promptSession = {
      id: prompt.id,
      n_prompt: ids.length,
      logit_max_abs_diff: logitDiff,
      matched_tokens: matched,
      total_tokens: expectedPrompt.generated_ids.length,
      first_mismatch: firstMismatch,
      near_tie: expectedPrompt.near_tie ?? false,
      check_tokens: check.tokens,
      runs,
      medians,
    };
    promptSessions.push(promptSession);
    addRow(promptSession);
  }

  const allRuns = promptSessions.flatMap((prompt) => prompt.runs);
  const sessionMedians = {
    decode_tok_s: median(allRuns.map((run) => run.decode_tok_s)),
    prefill_tok_s: median(allRuns.map((run) => run.prefill_tok_s)),
    ttft_ms: median(allRuns.map((run) => run.ttft_ms)),
  };

  const session = {
    label: `${target}-${WEIGHTS}`,
    model: expected.model,
    commit: "unknown",
    started_at: startedAt.toISOString(),
    target,
    backend: readyMessage.backend,
    requested_backend: readyMessage.requestedBackend,
    fallback: readyMessage.fallback,
    pipeline_depth: readyMessage.pipelineDepth,
    gpu: readyMessage.gpu,
    browser: BROWSER,
    pkg: readyMessage.pkg,
    threads: readyMessage.threads,
    simd: readyMessage.simd,
    weights: WEIGHTS,
    host: {
      nproc: navigator.hardwareConcurrency || 1,
      cpu: navigator.userAgent,
      memory_limit: navigator.deviceMemory
        ? `${navigator.deviceMemory} GB`
        : "unknown",
      kernel: navigator.platform || "unknown",
      user_agent: navigator.userAgent,
    },
    worker_cmd: [
      "browser",
      `backend=${BACKEND}`,
      `weights=${WEIGHTS}`,
      `pkg=${readyMessage.pkg}`,
      `threads=${readyMessage.threads}`,
      `pipeline=${readyMessage.pipelineDepth}`,
    ],
    load_ms: readyMessage.loadMs,
    peak_mb: null,
    warmup: WARMUP,
    runs: RUNS,
    prompts: promptSessions,
    session_medians: sessionMedians,
    matches_reference: matchesReference,
    near_tie_mismatch: nearTieMismatch,
  };

  summary.textContent = `${WEIGHTS} weights on ${target}, ${matchesReference ? "matches the reference" : "does not match the reference"}: ${sessionMedians.decode_tok_s.toFixed(2)} tokens/s decode, ${sessionMedians.prefill_tok_s.toFixed(1)} tokens/s prefill, ${sessionMedians.ttft_ms.toFixed(0)} ms to first token.`;
  statusLine.textContent = "Finished.";
  downloadButton.disabled = false;
  downloadButton.addEventListener("click", () => {
    const blob = new Blob([`${JSON.stringify(session, null, 2)}\n`], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `session-${target}-${WEIGHTS}.json`;
    link.click();
    URL.revokeObjectURL(url);
  });

  window.__benchResult = session;
  window.__benchDone = true;
}

function fail(message) {
  statusLine.textContent = `Benchmark failed: ${message}`;
  window.__benchError = message;
}

main().catch((error) => fail(error.message));
