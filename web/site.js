const REPO_URL = "https://github.com/pavanmanishd/gpt2-fast/blob/main/";
const REPO_TREE_URL = REPO_URL.replace("/blob/main/", "/tree/main/");
const MODEL_BASE_URL = "/models/gpt2/";
const CACHE_NAME = "gpt2-weights-v2";
const WEIGHTS = "int8";
const WEIGHTS_FILE = "model-int8.safetensors";
const THREAD_MILESTONE = /^N5-threads\d+$/;

const els = {
  loadPanel: document.getElementById("load-panel"),
  loadButton: document.getElementById("load-model"),
  loadStatus: document.getElementById("load-status"),
  progressWrap: document.getElementById("progress-wrap"),
  progress: document.getElementById("progress"),
  progressFill: document.getElementById("progress-fill"),
  prompt: document.getElementById("prompt"),
  generate: document.getElementById("generate"),
  stop: document.getElementById("stop"),
  maxTokens: document.getElementById("max-tokens"),
  output: document.getElementById("output"),
  meterSpeed: document.getElementById("meter-speed"),
  meterTtft: document.getElementById("meter-ttft"),
  meterTokens: document.getElementById("meter-tokens"),
  demoStatus: document.getElementById("demo-status"),
  simd: document.getElementById("simd-state"),
  threads: document.getElementById("threads-state"),
  deviceCopy: document.getElementById("device-copy"),
  runBench: document.getElementById("run-bench"),
  benchStatus: document.getElementById("bench-status"),
  benchResult: document.getElementById("bench-result"),
  deviceDecode: document.getElementById("device-decode"),
  devicePrefill: document.getElementById("device-prefill"),
  deviceTtft: document.getElementById("device-ttft"),
  projectDecode: document.getElementById("project-decode"),
  projectPrefill: document.getElementById("project-prefill"),
  projectTtft: document.getElementById("project-ttft"),
  projectNote: document.getElementById("project-note"),
  chart: document.getElementById("chart"),
  milestones: document.getElementById("milestones"),
  storyIntro: document.getElementById("story-intro"),
  storyNote: document.getElementById("story-note"),
};

const integerFormat = new Intl.NumberFormat("en-US");
const rateFormat1 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
const rateFormat2 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });
const rateFormat3 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 3 });

let worker = null;
let pending = new Map();
let nextId = 1;
let loadHandlers = null;
let probing = true;
let loading = false;
let ready = false;
let busy = false;
let generating = false;
let stopRequested = false;
let downloaded = false;
let deviceBench = null;
let lastTokenCount = 0;
let chartEntries = [];
let lastChartWidth = 0;
let chartFrame = 0;

function formatCount(value) {
  if (value === null || value === undefined) {
    return "\u2013";
  }
  const number = Number(value);
  return Number.isFinite(number) ? integerFormat.format(number) : "\u2013";
}

function formatRate(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) {
    return "\u2013";
  }
  const format =
    number < 1 ? rateFormat3 : number < 100 ? rateFormat2 : rateFormat1;
  return format.format(number);
}

function formatLiveRate(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) {
    return "\u2013";
  }
  return (number < 10 ? rateFormat2 : rateFormat1).format(number);
}

function formatMs(value) {
  if (value === null || value === undefined) {
    return "\u2013";
  }
  const number = Number(value);
  if (!Number.isFinite(number) || number < 0) {
    return "\u2013";
  }
  if (number >= 1000) {
    return `${(number / 1000).toFixed(2)} s`;
  }
  return `${Math.round(number)} ms`;
}

function formatLoadTime(value) {
  if (value === null || value === undefined) {
    return "\u2013";
  }
  const number = Number(value);
  if (!Number.isFinite(number) || number < 0) {
    return "\u2013";
  }
  if (number >= 10000) {
    return `${Math.round(number / 1000)} s`;
  }
  return `${(number / 1000).toFixed(1)} s`;
}

function formatMemory(value) {
  if (value === null || value === undefined) {
    return "\u2013";
  }
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "\u2013";
  }
  return `${integerFormat.format(Math.round(number))} MB`;
}

function formatBytes(bytes) {
  if (bytes === null || bytes === undefined) {
    return "\u2013";
  }
  const number = Number(bytes);
  if (!Number.isFinite(number) || number < 0) {
    return "\u2013";
  }
  if (number >= 1e6) {
    return `${(number / 1e6).toFixed(number >= 10e6 ? 0 : 1)} MB`;
  }
  if (number >= 1e3) {
    return `${Math.round(number / 1e3)} kB`;
  }
  return `${number} B`;
}

function formatProtocol(key, value) {
  if (value === null || value === undefined) {
    return "\u2013";
  }
  if (key === "near_tie_min") {
    return Number(value).toFixed(2);
  }
  if (key === "closest_gap") {
    return Number(value).toFixed(3);
  }
  return formatCount(value);
}

function describeError(error) {
  return error instanceof Error ? error.message : String(error);
}

function isComparisonEngine(entry) {
  const id = String(entry.milestone);
  return id.startsWith("R0-") || (id.startsWith("N9-") && !/gpt2/i.test(id));
}

function isQuantized(entry) {
  return (
    /(?:^|[-_])q8(?:_|$)|int8/i.test(String(entry.milestone)) ||
    (entry.perplexity_f32 !== null && entry.perplexity_f32 !== undefined)
  );
}

function escapeXml(value) {
  return String(value).replace(
    /[&<>"']/g,
    (character) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&apos;",
      })[character],
  );
}

function updateProgress(percent) {
  const value = Math.max(0, Math.min(100, Math.round(percent)));
  els.progressFill.style.width = `${value}%`;
  els.progress.setAttribute("aria-valuenow", String(value));
}

function updateControls() {
  els.loadButton.hidden = probing || loading || ready;
  els.loadButton.disabled = loading;
  els.progressWrap.hidden = !loading;
  els.loadPanel.hidden = ready;
  els.generate.disabled = !ready || busy;
  els.stop.disabled = !generating;
  els.prompt.disabled = busy;
  els.maxTokens.disabled = busy;
  els.runBench.disabled = !ready || busy || !deviceBench;
}

async function modelIsCached() {
  if (!("caches" in window)) {
    return false;
  }
  const url = new URL(WEIGHTS_FILE, new URL(MODEL_BASE_URL, location.origin));
  url.searchParams.set("part", "manifest");
  const cache = await caches.open(CACHE_NAME);
  return Boolean(await cache.match(url.href));
}

function createWorker() {
  const threads = preferredThreads();
  const pkg = self.crossOriginIsolated && threads > 1 ? "pkg-threads" : "pkg";
  const instance = new Worker(
    new URL(
      `./worker.js?pkg=${encodeURIComponent(pkg)}&threads=${threads}`,
      import.meta.url,
    ),
    { type: "module" },
  );
  instance.onmessage = (event) => handleMessage(instance, event.data);
  instance.onerror = (event) =>
    handleWorkerFailure(instance, event.message || "The worker failed.");
  return instance;
}

const THREAD_CAP = 8;

function preferredThreads() {
  const requested = Number.parseInt(
    new URLSearchParams(location.search).get("threads") ?? "",
    10,
  );
  if (Number.isFinite(requested) && requested > 0) {
    return Math.min(requested, 32);
  }
  return Math.min(navigator.hardwareConcurrency || 1, THREAD_CAP);
}

function handleMessage(instance, message) {
  if (instance !== worker) {
    return;
  }
  if (message.type === "progress") {
    handleProgress(message);
    return;
  }
  if (message.type === "ready") {
    if (loadHandlers) {
      const handlers = loadHandlers;
      loadHandlers = null;
      handlers.resolve(message);
    }
    return;
  }
  if (message.type === "error" && message.id === undefined) {
    handleGlobalError(message.message);
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
}

function handleProgress(message) {
  const filePart =
    message.total > 0 ? Math.min(1, message.loaded / message.total) : 0;
  const overall = Math.round(
    ((message.index - 1 + filePart) / message.count) * 100,
  );
  updateProgress(overall);
  if (!message.cached) {
    downloaded = true;
  }
  const size = message.total
    ? `${formatBytes(message.loaded)} of ${formatBytes(message.total)}`
    : formatBytes(message.loaded);
  const action = message.cached ? "Reading from cache" : "Downloading";
  els.loadStatus.textContent = `${action} ${message.file} (${overall}%, ${size}).`;
}

function handleGlobalError(message) {
  if (loadHandlers) {
    const handlers = loadHandlers;
    loadHandlers = null;
    handlers.reject(new Error(message));
    if (worker) {
      worker.terminate();
      worker = null;
    }
    return;
  }
  handleWorkerFailure(worker, message);
}

function handleWorkerFailure(instance, message) {
  if (instance !== worker) {
    return;
  }
  const error = new Error(message);
  rejectAll(error);
  if (worker) {
    worker.terminate();
    worker = null;
  }
  if (loadHandlers) {
    const handlers = loadHandlers;
    loadHandlers = null;
    handlers.reject(error);
  }
  if (loading) {
    return;
  }
  ready = false;
  busy = false;
  generating = false;
  els.loadPanel.hidden = false;
  els.loadButton.hidden = false;
  els.progressWrap.hidden = true;
  els.loadStatus.textContent = `The worker stopped: ${message}`;
  els.demoStatus.textContent = "The model is not loaded.";
  els.benchStatus.textContent = "Load the model above first.";
  els.simd.textContent = "not loaded";
  els.threads.textContent = "not loaded";
  updateControls();
}

function rejectAll(error) {
  for (const request of pending.values()) {
    request.reject(error);
  }
  pending.clear();
}

function request(message, onToken) {
  if (!worker) {
    return Promise.reject(new Error("The worker is not running."));
  }
  const id = nextId;
  nextId += 1;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject, onToken });
    worker.postMessage({ ...message, id });
  });
}

async function loadModel() {
  if (loading || ready) {
    return;
  }
  loading = true;
  downloaded = false;
  updateProgress(0);
  els.loadPanel.hidden = false;
  els.loadStatus.textContent = "Starting the worker.";
  els.demoStatus.textContent = "Loading the model.";
  updateControls();

  try {
    if (!worker) {
      worker = createWorker();
    }
    const message = await new Promise((resolve, reject) => {
      loadHandlers = { resolve, reject };
      worker.postMessage({ type: "load", weights: WEIGHTS });
    });
    ready = true;
    loading = false;
    els.loadPanel.hidden = true;
    els.progressWrap.hidden = true;
    els.simd.textContent = message.simd ? "on" : "off";
    els.threads.textContent =
      message.threads > 1
        ? `${message.threads} threads`
        : `off (1 thread)`;
    els.demoStatus.textContent = `Model ready in ${formatLoadTime(message.loadMs)}${downloaded ? ", downloaded and cached" : ", from browser cache"}.`;
    if (deviceBench) {
      els.benchStatus.textContent = "Ready to run.";
    }
    updateControls();
  } catch (error) {
    loadHandlers = null;
    loading = false;
    ready = false;
    if (worker) {
      worker.terminate();
      worker = null;
    }
    els.loadPanel.hidden = false;
    els.loadButton.hidden = false;
    els.progressWrap.hidden = true;
    els.loadStatus.textContent = `Could not load the model: ${describeError(error)}`;
    els.demoStatus.textContent = "The model is not loaded.";
    els.benchStatus.textContent = "Load the model above first.";
    els.simd.textContent = "not loaded";
    els.threads.textContent = "not loaded";
    updateControls();
  }
}

function stopGeneration() {
  if (!generating) {
    return;
  }
  stopRequested = true;
  if (worker) {
    worker.terminate();
    worker = null;
  }
  rejectAll(new Error("Stopped"));
  ready = false;
  loading = false;
  busy = false;
  generating = false;
  els.loadPanel.hidden = false;
  els.loadButton.hidden = false;
  els.progressWrap.hidden = true;
  els.loadStatus.textContent = "Stopped. Load the model again to continue.";
  els.demoStatus.textContent =
    lastTokenCount > 0
      ? `Stopped after ${formatCount(lastTokenCount)} tokens.`
      : "Stopped before the first token.";
  els.benchStatus.textContent = "Load the model above first.";
  els.simd.textContent = "not loaded";
  els.threads.textContent = "not loaded";
  updateControls();
}

async function generate() {
  if (!ready || busy) {
    return;
  }
  const text = els.prompt.value;
  if (text.length === 0) {
    els.demoStatus.textContent = "Enter a prompt first.";
    return;
  }

  busy = true;
  generating = true;
  stopRequested = false;
  lastTokenCount = 0;
  els.output.textContent = "";
  els.meterSpeed.textContent = "\u2013";
  els.meterTtft.textContent = "\u2013";
  els.meterTokens.textContent = "0";
  els.demoStatus.textContent = "Encoding the prompt.";
  updateControls();

  let count = 0;
  let firstAt = 0;

  try {
    const encoded = await request({ type: "encode", text });
    if (!encoded.ids || encoded.ids.length === 0) {
      throw new Error("The prompt produced no tokens.");
    }
    els.demoStatus.textContent = "Prefilling the prompt.";
    const startedAt = performance.now();
    const done = await request(
      {
        type: "generate",
        ids: encoded.ids,
        maxNewTokens: Number(els.maxTokens.value),
        stream: true,
      },
      (message) => {
        const now = performance.now();
        if (count === 0) {
          firstAt = now;
          els.meterTtft.textContent = formatMs(now - startedAt);
        }
        count += 1;
        lastTokenCount = count;
        els.meterTokens.textContent = formatCount(count);
        if (count > 1 && now > firstAt) {
          els.meterSpeed.textContent = formatLiveRate(
            (count - 1) / ((now - firstAt) / 1000),
          );
        }
        const nearBottom =
          els.output.scrollHeight -
            els.output.scrollTop -
            els.output.clientHeight <
          40;
        els.output.textContent = message.text;
        if (nearBottom) {
          els.output.scrollTop = els.output.scrollHeight;
        }
      },
    );

    const timings = Array.isArray(done.timings) ? done.timings : [];
    const tokenCount = Array.isArray(done.tokens) ? done.tokens.length : count;
    lastTokenCount = tokenCount;
    els.meterTokens.textContent = formatCount(tokenCount);
    if (timings.length > 0) {
      els.meterTtft.textContent = formatMs(timings[0]);
      if (timings.length > 1) {
        const seconds = (timings[timings.length - 1] - timings[0]) / 1000;
        if (seconds > 0) {
          els.meterSpeed.textContent = formatRate(
            (tokenCount - 1) / seconds,
          );
        }
      }
    }
    const speed = els.meterSpeed.textContent;
    els.demoStatus.textContent = `Generated ${formatCount(tokenCount)} tokens${
      speed !== "\u2013" ? ` at ${speed} tokens/s` : ""
    }.`;
  } catch (error) {
    if (!stopRequested) {
      els.demoStatus.textContent = `Generation failed: ${describeError(error)}`;
    }
  } finally {
    busy = false;
    generating = false;
    stopRequested = false;
    updateControls();
  }
}

async function runBenchmark() {
  if (!ready || busy || !deviceBench) {
    return;
  }
  busy = true;
  els.benchResult.hidden = true;
  els.benchStatus.textContent = "Running one warm-up.";
  updateControls();

  try {
    const encoded = await request({
      type: "encode",
      text: deviceBench.prompt,
    });
    if (!encoded.ids || encoded.ids.length === 0) {
      throw new Error("The prompt produced no tokens.");
    }
    await request({
      type: "generate",
      ids: encoded.ids,
      maxNewTokens: Number(deviceBench.max_new_tokens),
      stream: false,
    });

    els.benchStatus.textContent = "Running the timed run.";
    const done = await request({
      type: "generate",
      ids: encoded.ids,
      maxNewTokens: Number(deviceBench.max_new_tokens),
      stream: false,
    });
    const timings = Array.isArray(done.timings) ? done.timings : [];
    const tokenCount = Array.isArray(done.tokens)
      ? done.tokens.length
      : Number(deviceBench.max_new_tokens);
    if (timings.length === 0) {
      throw new Error("The worker returned no timings.");
    }

    const ttft = timings[0];
    const decodeSeconds =
      timings.length > 1
        ? (timings[timings.length - 1] - timings[0]) / 1000
        : 0;
    const decode =
      decodeSeconds > 0 ? (tokenCount - 1) / decodeSeconds : 0;
    const prefill = ttft > 0 ? encoded.ids.length / (ttft / 1000) : 0;

    els.deviceDecode.textContent = formatRate(decode);
    els.devicePrefill.textContent = formatRate(prefill);
    els.deviceTtft.textContent = formatMs(ttft);
    els.benchResult.hidden = false;
    els.benchStatus.textContent = `Finished: ${formatCount(
      tokenCount,
    )} new tokens after one warm-up.`;
  } catch (error) {
    els.benchStatus.textContent = `Benchmark failed: ${describeError(error)}`;
  } finally {
    busy = false;
    updateControls();
  }
}

function appendDelta(list, label, before, after, unit, formatter) {
  const item = document.createElement("li");
  const labelNode = document.createElement("span");
  labelNode.className = "delta-label";
  labelNode.textContent = label;

  const valueNode = document.createElement("span");
  valueNode.className = "delta-values";

  if (before !== null && before !== undefined && Number.isFinite(Number(before))) {
    const beforeNode = document.createElement("span");
    beforeNode.className = "delta-before";
    beforeNode.textContent = formatter(before);
    const arrow = document.createElement("span");
    arrow.className = "arrow";
    arrow.textContent = "\u2192";
    const afterNode = document.createElement("strong");
    afterNode.textContent = formatter(after);
    valueNode.append(beforeNode, " ", arrow, " ", afterNode);
  } else {
    const afterNode = document.createElement("strong");
    afterNode.textContent = formatter(after);
    valueNode.append(afterNode);
  }

  if (unit) {
    valueNode.append(` ${unit}`);
  }
  item.append(labelNode, valueNode);
  list.append(item);
}

function correctnessText(entry, protocol) {
  const total = Number(protocol.tokens_checked);
  const tokenText =
    Number.isFinite(total) && total > 0 ? formatCount(total) : "the reference";
  if (entry.milestone === "R0-torch") {
    return "The reference run. Every engine is checked against these tokens.";
  }
  if (entry.matches_reference) {
    return `Matches the reference on all ${tokenText} tokens.`;
  }
  const rate = Number(entry.token_match_rate);
  const quantized = isQuantized(entry);
  if (
    quantized &&
    entry.perplexity !== null &&
    entry.perplexity !== undefined &&
    entry.perplexity_f32 !== null &&
    entry.perplexity_f32 !== undefined
  ) {
    const delta =
      (Number(entry.perplexity) / Number(entry.perplexity_f32) - 1) * 100;
    const sign = delta >= 0 ? "+" : "";
    const matched =
      Number.isFinite(total) && total > 0 && Number.isFinite(rate)
        ? ` (${formatCount(Math.round(rate * total))} of ${tokenText})`
        : "";
    const percent = Number.isFinite(rate) ? Math.round(rate * 100) : "?";
    return `int8: ${percent}% of tokens match${matched}, perplexity ${sign}${delta.toFixed(
      1,
    )}% (${Number(entry.perplexity).toFixed(2)} vs ${Number(
      entry.perplexity_f32,
    ).toFixed(2)} f32).`;
  }
  if (Number.isFinite(rate)) {
    const matched =
      Number.isFinite(total) && total > 0
        ? ` (${formatCount(Math.round(rate * total))} of ${tokenText})`
        : "";
    if (rate >= 1) {
      return `All ${tokenText} tokens match; the run misses the project's logit tolerance, so it is not an exact match.`;
    }
    if (quantized) {
      return `int8: ${Math.round(rate * 100)}% of tokens match${matched}.`;
    }
    return `${Math.round(rate * 100)}% of tokens match${matched}.`;
  }
  return "This milestone does not match the reference exactly.";
}

function buildThreadTable(entry, scaling, protocol) {
  const rows = scaling.map((item) => ({
    threads: Number(String(item.milestone).replace("N5-threads", "")),
    item,
  }));
  const cores = Number(protocol.cores);
  if (Number.isFinite(cores) && cores > 0) {
    rows.push({ threads: cores, item: entry });
  }
  rows.sort((left, right) => left.threads - right.threads);

  const table = document.createElement("table");
  table.className = "thread-table";
  const caption = document.createElement("caption");
  caption.textContent = `Thread scaling: ${formatCount(
    rows[0].threads,
  )} to ${formatCount(rows[rows.length - 1].threads)} threads.`;
  table.append(caption);

  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const label of ["Threads", "Decode tok/s", "Prefill tok/s"]) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = label;
    headRow.append(cell);
  }
  head.append(headRow);
  table.append(head);

  const body = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    const threadCell = document.createElement("th");
    threadCell.scope = "row";
    threadCell.textContent = formatCount(row.threads);
    tr.append(threadCell);
    for (const value of [row.item.decode_tok_s, row.item.prefill_tok_s]) {
      const cell = document.createElement("td");
      cell.textContent = formatRate(value);
      tr.append(cell);
    }
    body.append(tr);
  }
  table.append(body);
  return table;
}

function renderCards(entries, scaling, protocol) {
  els.milestones.replaceChildren();
  let previous = null;

  for (const entry of entries) {
    const article = document.createElement("article");
    article.className = "milestone";

    const meta = document.createElement("p");
    meta.className = "milestone-meta";
    const id = document.createElement("span");
    id.className = "milestone-id";
    id.textContent = entry.milestone;
    meta.append(id);

    const metaParts = [entry.date, entry.commit, entry.target].filter(Boolean);
    if (
      entry.sessions !== undefined &&
      Number(entry.sessions) !== Number(protocol.sessions)
    ) {
      metaParts.push(
        `${formatCount(entry.sessions)} session${
          Number(entry.sessions) === 1 ? "" : "s"
        }`,
      );
    }
    for (const part of metaParts) {
      const node = document.createElement("span");
      node.textContent = part;
      meta.append(node);
    }
    if (isQuantized(entry)) {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = "int8";
      meta.append(tag);
    }

    const title = document.createElement("h3");
    title.textContent = entry.title;

    const headline = document.createElement("p");
    headline.className = "milestone-headline";
    headline.textContent = entry.headline;

    const deltas = document.createElement("ul");
    deltas.className = "deltas";
    const isComparison = isComparisonEngine(entry);
    const showBefore =
      !isComparison && entry.milestone !== "N1" && previous !== null;
    appendDelta(
      deltas,
      "Decode",
      showBefore ? previous.decode_tok_s : null,
      entry.decode_tok_s,
      "tok/s",
      formatRate,
    );
    appendDelta(
      deltas,
      "Prefill",
      showBefore ? previous.prefill_tok_s : null,
      entry.prefill_tok_s,
      "tok/s",
      formatRate,
    );
    appendDelta(
      deltas,
      "First token",
      showBefore ? previous.ttft_ms : null,
      entry.ttft_ms,
      "",
      formatMs,
    );
    appendDelta(
      deltas,
      "Peak memory",
      showBefore ? previous.peak_mb : null,
      entry.peak_mb,
      "",
      formatMemory,
    );

    const correctness = document.createElement("p");
    correctness.className = "correctness";
    correctness.textContent = correctnessText(entry, protocol);

    article.append(meta, title, headline, deltas, correctness);

    if (entry.milestone === "N5" && scaling.length > 0) {
      article.append(buildThreadTable(entry, scaling, protocol));
    }

    if (entry.journal) {
      const link = document.createElement("a");
      link.className = "journal-link";
      link.href = REPO_URL + entry.journal;
      link.textContent = "Journal note";
      article.append(link);
    }

    els.milestones.append(article);
    if (!isComparison) {
      previous = entry;
    }
  }
}

function renderChart(entries) {
  if (entries.length === 0) {
    els.chart.replaceChildren();
    return;
  }
  const width = Math.max(240, Math.floor(els.chart.clientWidth || 600));
  lastChartWidth = width;
  const narrow = width < 430;
  const longestLabel = entries.reduce(
    (longest, entry) => Math.max(longest, String(entry.milestone).length),
    0,
  );
  const valueWidth = narrow ? 78 : 88;
  const labelWidth = narrow
    ? Math.max(
        58,
        Math.min(
          Math.round(width * 0.38),
          width - valueWidth - 80,
          longestLabel * 6.6 + 8,
        ),
      )
    : Math.max(
        68,
        Math.min(140, width - valueWidth - 80, longestLabel * 7.2 + 10),
      );
  const rowHeight = narrow ? 30 : 32;
  const barHeight = narrow ? 14 : 16;
  const top = 20;
  const axisHeight = 28;
  const left = labelWidth;
  const right = width - valueWidth;
  const plotWidth = Math.max(80, right - left);
  const height = top + entries.length * rowHeight + axisHeight;

  const values = entries
    .map((entry) => Number(entry.decode_tok_s))
    .filter((value) => Number.isFinite(value) && value > 0);
  if (values.length === 0) {
    els.chart.replaceChildren();
    return;
  }

  const minLog = Math.floor(Math.log10(Math.min(...values)));
  const maxLog = Math.ceil(Math.log10(Math.max(...values)) * 10) / 10;
  const logSpan = Math.max(0.1, maxLog - minLog);
  const xFor = (value) =>
    left + ((Math.log10(value) - minLog) / logSpan) * plotWidth;

  const parts = [
    `<svg class="chart-svg${
      narrow ? " chart-svg--narrow" : ""
    }" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" aria-hidden="true" focusable="false">`,
    '<defs><pattern id="int8-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect class="chart-hatch-bg" width="6" height="6"/><line class="chart-hatch-line" x1="0" y1="0" x2="0" y2="6"/></pattern></defs>',
  ];

  for (let exponent = minLog; exponent <= Math.floor(maxLog); exponent += 1) {
    const value = 10 ** exponent;
    const x = xFor(value);
    parts.push(
      `<line class="chart-grid" x1="${x.toFixed(1)}" y1="${
        top - 6
      }" x2="${x.toFixed(1)}" y2="${top + entries.length * rowHeight}"/>`,
    );
    const anchor = x < left + 12 ? "start" : x > right - 12 ? "end" : "middle";
    parts.push(
      `<text class="chart-tick" x="${x.toFixed(1)}" y="${
        top + entries.length * rowHeight + 16
      }" text-anchor="${anchor}">${value}</text>`,
    );
  }

  entries.forEach((entry, index) => {
    const value = Number(entry.decode_tok_s);
    const y = top + index * rowHeight + (rowHeight - barHeight) / 2;
    const rowCenter = y + barHeight / 2 + 4;
    const quantized = isQuantized(entry);
    const comparison = isComparisonEngine(entry);
    const barWidth =
      Number.isFinite(value) && value > 0
        ? Math.max(2, xFor(value) - left)
        : 0;
    const barClass = quantized
      ? "chart-bar-int8"
      : comparison
        ? "chart-bar-ref"
        : "chart-bar-ours";
    parts.push(
      `<text class="chart-label" x="2" y="${rowCenter.toFixed(
        1,
      )}">${escapeXml(entry.milestone)}</text>`,
    );
    if (barWidth > 0) {
      parts.push(
        `<rect class="${barClass}" x="${left}" y="${y.toFixed(
          1,
        )}" width="${barWidth.toFixed(1)}" height="${barHeight}" rx="2"/>`,
      );
    }
    const int8 = quantized
      ? '<tspan class="chart-int8" dx="4">int8</tspan>'
      : "";
    parts.push(
      `<text class="chart-value" x="${(left + barWidth + 6).toFixed(
        1,
      )}" y="${rowCenter.toFixed(1)}">${escapeXml(
        formatRate(value),
      )}${int8}</text>`,
    );
  });

  parts.push("</svg>");
  els.chart.innerHTML = parts.join("");
}

function scheduleChart() {
  if (chartEntries.length === 0) {
    return;
  }
  cancelAnimationFrame(chartFrame);
  chartFrame = requestAnimationFrame(() => {
    const width = Math.max(240, Math.floor(els.chart.clientWidth || 600));
    if (width !== lastChartWidth) {
      renderChart(chartEntries);
    }
  });
}

function renderStory(payload) {
  const protocol = payload.protocol ?? {};
  const milestones = payload.milestones ?? [];
  const main = milestones.filter(
    (entry) => !THREAD_MILESTONE.test(String(entry.milestone)),
  );
  const scaling = milestones.filter((entry) =>
    THREAD_MILESTONE.test(String(entry.milestone)),
  );

  chartEntries = main;
  renderChart(main);
  renderCards(main, scaling, protocol);

  const n1 = milestones.find((entry) => entry.milestone === "N1");
  if (
    n1 &&
    Number.isFinite(Number(n1.sessions)) &&
    Number(n1.sessions) !== Number(protocol.sessions)
  ) {
    els.storyNote.textContent = `N1 used ${formatCount(
      n1.sessions,
    )} session${
      Number(n1.sessions) === 1 ? "" : "s"
    } and no warm-up. From N2 on, each milestone uses ${formatCount(
      protocol.sessions,
    )} sessions.`;
  } else {
    els.storyNote.textContent = "";
  }
}

function renderDevice(protocol, bench, milestone) {
  const weights = milestone && isQuantized(milestone) ? "int8" : "f32";
  if (!bench) {
    els.deviceCopy.textContent =
      "The benchmark configuration could not be loaded.";
  } else {
    els.deviceCopy.textContent = `Run prompt ${bench.prompt_id} with ${formatCount(
      bench.max_new_tokens,
    )} new tokens: ${formatCount(bench.warmup)} warm-up${
      Number(bench.warmup) === 1 ? "" : "s"
    }, then one timed run. This page uses int8 weights in WebAssembly; the project column is the same ${weights} engine in a Linux container limited to ${formatCount(
      protocol.cores,
    )} cores.`;
  }

  if (!milestone) {
    els.projectNote.textContent = "";
    return;
  }

  els.projectDecode.textContent = formatRate(milestone.decode_tok_s);
  els.projectPrefill.textContent = formatRate(milestone.prefill_tok_s);
  els.projectTtft.textContent = formatMs(milestone.ttft_ms);
  els.projectNote.textContent = `Project: ${
    milestone.milestone
  }, ${weights} weights, ${formatCount(protocol.cores)} cores, Linux container. Numbers from this tab are measured in the browser.`;
}

function linkRepo() {
  for (const link of document.querySelectorAll("[data-repo]")) {
    link.href = REPO_URL + link.dataset.repo;
  }
  for (const link of document.querySelectorAll("[data-tree]")) {
    link.href = REPO_TREE_URL + link.dataset.tree;
  }
}

function renderData(payload) {
  const protocol = payload.protocol ?? {};
  for (const element of document.querySelectorAll("[data-protocol]")) {
    element.textContent = formatProtocol(
      element.dataset.protocol,
      protocol[element.dataset.protocol],
    );
  }

  const milestones = payload.milestones ?? [];
  deviceBench = payload.device_bench ?? null;
  const projectMilestone =
    milestones.find((entry) => entry.milestone === "N7") ??
    [...milestones].reverse().find(isQuantized) ??
    milestones[milestones.length - 1] ??
    null;

  renderDevice(protocol, deviceBench, projectMilestone);
  renderStory(payload);
  if (ready && deviceBench) {
    els.benchStatus.textContent = "Ready to run.";
  }
  updateControls();
}

function renderDataError(error) {
  console.error(error);
  els.storyIntro.textContent = "The milestone data could not be loaded.";
  const message = document.createElement("p");
  message.className = "status";
  message.textContent = "The milestone data is unavailable.";
  els.milestones.replaceChildren(message);
  els.chart.replaceChildren();
  els.storyNote.textContent = "";
  els.deviceCopy.textContent = "The benchmark configuration could not be loaded.";
  els.benchStatus.textContent = "The benchmark is unavailable.";
  for (const element of document.querySelectorAll("[data-protocol]")) {
    element.textContent = "\u2013";
  }
  updateControls();
}

async function fetchData() {
  const response = await fetch("./data/milestones.json", {
    cache: "no-cache",
  });
  if (!response.ok) {
    throw new Error(`data/milestones.json: HTTP ${response.status}`);
  }
  return response.json();
}

function wireEvents() {
  els.loadButton.addEventListener("click", loadModel);
  els.generate.addEventListener("click", generate);
  els.stop.addEventListener("click", stopGeneration);
  els.runBench.addEventListener("click", runBenchmark);
  window.addEventListener("resize", scheduleChart);
}

async function init() {
  wireEvents();
  linkRepo();

  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver(() => {
      if (chartEntries.length === 0) {
        return;
      }
      const width = Math.max(
        240,
        Math.floor(els.chart.clientWidth || 600),
      );
      if (width !== lastChartWidth) {
        renderChart(chartEntries);
      }
    });
    observer.observe(els.chart);
  }

  fetchData().then(renderData).catch(renderDataError);

  const cached = await modelIsCached().catch(() => false);
  probing = false;
  if (cached) {
    els.loadStatus.textContent = "The model is cached. Loading it.";
    loadModel();
  } else {
    els.loadButton.hidden = false;
    els.loadStatus.textContent =
      "The model is not loaded. It downloads once and stays in this browser's cache.";
    updateControls();
  }
}

init();
