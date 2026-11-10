const REPO_URL = "https://github.com/pavanmanishd/gpt2-fast/blob/main/";
const REPO_TREE_URL = REPO_URL.replace("/blob/main/", "/tree/main/");
const MODEL_BASE_URL = "/models/gpt2/";
const CACHE_NAME = "gpt2-weights-v2";
const WEIGHTS = "int8";
const WEIGHTS_FILE = "model-int8.safetensors";
const THREAD_CAP = 8;
const DEFAULT_DATA_URL = "./data/milestones.json";

const SIMD_SUPPORTED = detectSimd();

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
  backendMode: document.getElementById("backend-mode"),
  output: document.getElementById("output"),
  meterSpeed: document.getElementById("meter-speed"),
  meterTtft: document.getElementById("meter-ttft"),
  meterTokens: document.getElementById("meter-tokens"),
  demoStatus: document.getElementById("demo-status"),
  running: document.getElementById("running-state"),
  weights: document.getElementById("weights-state"),
  cache: document.getElementById("cache-state"),
  backendNote: document.getElementById("backend-note"),
  deviceCopy: document.getElementById("device-copy"),
  runBench: document.getElementById("run-bench"),
  benchStatus: document.getElementById("bench-status"),
  benchTableWrap: document.getElementById("bench-table-wrap"),
  benchBody: document.getElementById("bench-body"),
  benchNote: document.getElementById("bench-note"),
  benchHint: document.getElementById("bench-hint"),
  story: document.getElementById("story"),
  storyIntro: document.getElementById("story-intro"),
  storyNative: document.getElementById("story-native"),
  storyRace: document.getElementById("story-race"),
  storySizes: document.getElementById("story-sizes"),
  storyWeb: document.getElementById("story-web"),
  storyNote: document.getElementById("story-note"),
  nativeChart: document.getElementById("native-chart"),
  nativeMilestones: document.getElementById("native-milestones"),
  raceTables: document.getElementById("race-tables"),
  sizesBody: document.getElementById("sizes-body"),
  webChart: document.getElementById("web-chart"),
  webMilestones: document.getElementById("web-milestones"),
};

const integerFormat = new Intl.NumberFormat("en-US");
const rateFormat1 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
const rateFormat2 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });
const rateFormat3 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 3 });

let pageClient = null;
let modelReady = null;
let probing = true;
let loading = false;
let generating = false;
let busy = false;
let stopRequested = false;
let downloaded = false;
let lastTokenCount = 0;
let backendMode = "auto";
let payloadData = null;
let deviceBench = null;
let benchRunning = false;
let benchResults = [];

const chartData = new Map();
const chartWidths = new Map();

function detectSimd() {
  try {
    return WebAssembly.validate(
      new Uint8Array([
        0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10,
        10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
      ]),
    );
  } catch {
    return false;
  }
}

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

function formatPerplexity(value) {
  if (value === null || value === undefined) {
    return "\u2013";
  }
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(2) : "\u2013";
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

function isReference(entry) {
  return String(entry.milestone).startsWith("R0");
}

function isQuantized(entry) {
  if (entry.quantized !== undefined) {
    return Boolean(entry.quantized);
  }
  const milestone = String(entry.milestone ?? "");
  return milestone.endsWith("-int8") || milestone === "N7";
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

function workerPackage() {
  if (!SIMD_SUPPORTED) {
    return "pkg-nosimd";
  }
  return self.crossOriginIsolated && preferredThreads() > 1
    ? "pkg-threads"
    : "pkg";
}

function requestedBackend() {
  return backendMode === "cpu" ? "cpu" : "webgpu";
}

function workerUrl(requested) {
  const params = new URLSearchParams({
    backend: requested,
    pkg: workerPackage(),
    threads: String(preferredThreads()),
  });
  return `./worker.js?${params.toString()}`;
}

class EngineClient {
  constructor(requested, options = {}) {
    this.requested = requested;
    this.onProgress = options.onProgress ?? null;
    this.onError = options.onError ?? null;
    this.pending = new Map();
    this.nextId = 1;
    this.loadWaiter = null;
    this.closed = false;
    this.worker = new Worker(new URL(workerUrl(requested), import.meta.url), {
      type: "module",
    });
    this.worker.onmessage = (event) => this.handle(event.data);
    this.worker.onerror = (event) =>
      this.fail(event.message || "The worker failed.");
  }

  load(weights) {
    if (this.closed) {
      return Promise.reject(new Error("The worker is not running."));
    }
    return new Promise((resolve, reject) => {
      this.loadWaiter = { resolve, reject };
      this.worker.postMessage({ type: "load", weights });
    });
  }

  request(message, onToken) {
    if (this.closed) {
      return Promise.reject(new Error("The worker is not running."));
    }
    const id = this.nextId;
    this.nextId += 1;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onToken });
      this.worker.postMessage({ ...message, id });
    });
  }

  terminate(message = "Stopped") {
    if (this.closed) {
      return;
    }
    this.closed = true;
    this.worker.terminate();
    const error = new Error(message);
    if (this.loadWaiter) {
      this.loadWaiter.reject(error);
      this.loadWaiter = null;
    }
    for (const request of this.pending.values()) {
      request.reject(error);
    }
    this.pending.clear();
  }

  handle(message) {
    if (this.closed) {
      return;
    }
    if (message.type === "progress") {
      this.onProgress?.(message);
      return;
    }
    if (message.type === "ready") {
      if (this.loadWaiter) {
        const waiter = this.loadWaiter;
        this.loadWaiter = null;
        waiter.resolve(message);
      }
      return;
    }
    if (message.type === "error" && message.id === undefined) {
      const waiter = this.loadWaiter;
      this.loadWaiter = null;
      if (waiter) {
        waiter.reject(new Error(message.message));
      } else {
        this.onError?.(message.message);
      }
      return;
    }

    const request = this.pending.get(message.id);
    if (!request) {
      return;
    }
    if (message.type === "token") {
      request.onToken?.(message);
      return;
    }
    this.pending.delete(message.id);
    if (message.type === "error") {
      request.reject(new Error(message.message));
    } else {
      request.resolve(message);
    }
  }

  fail(message) {
    if (this.closed) {
      return;
    }
    this.closed = true;
    this.worker.terminate();
    const error = new Error(message);
    const waiter = this.loadWaiter;
    this.loadWaiter = null;
    for (const request of this.pending.values()) {
      request.reject(error);
    }
    this.pending.clear();
    if (waiter) {
      waiter.reject(error);
    } else {
      this.onError?.(message);
    }
  }
}

function describeEngine(ready) {
  if (!ready) {
    return "not loaded";
  }
  if (ready.backend === "webgpu") {
    const adapter = ready.gpu?.adapter ?? {};
    const name = [adapter.vendor, adapter.architecture]
      .filter(Boolean)
      .join(" ");
    const detail =
      name || adapter.description || adapter.device || "adapter not named";
    return `WebGPU \u00b7 ${detail}`;
  }
  const threads = Number(ready.threads) || 1;
  const simd = ready.simd ? "SIMD" : "scalar";
  if (threads > 1) {
    return `WebAssembly \u00b7 ${threads} threads \u00b7 ${simd}`;
  }
  return `WebAssembly \u00b7 ${simd} \u00b7 1 thread`;
}

function weightsLabel() {
  const weights = payloadData?.weights;
  if (!weights) {
    return "int8";
  }
  const size = Number(weights.bytes);
  const sizeText = Number.isFinite(size) ? `, ${formatBytes(size)}` : "";
  return `${weights.format}${sizeText}`;
}

function updateLoadButton() {
  const size = payloadData?.weights ? `${weightsLabel()}, ` : "";
  els.loadButton.textContent = `Load the model (${size}cached after the first time)`;
}

function updateProgress(percent) {
  const value = Math.max(0, Math.min(100, Math.round(percent)));
  els.progressFill.style.width = `${value}%`;
  els.progress.setAttribute("aria-valuenow", String(value));
}

function updateControls() {
  els.loadButton.hidden = probing || loading || modelReady !== null;
  els.loadButton.disabled = loading;
  els.progressWrap.hidden = !loading;
  els.loadPanel.hidden = modelReady !== null;
  els.generate.disabled = modelReady === null || busy;
  els.stop.disabled = !generating;
  els.prompt.disabled = busy;
  els.maxTokens.disabled = busy;
  els.backendMode.disabled = loading || busy;
  els.runBench.disabled =
    modelReady === null || busy || benchRunning || !deviceBench;
}

function setRunningState() {
  els.running.textContent = describeEngine(modelReady);
  els.weights.textContent = weightsLabel();
  if (modelReady === null) {
    els.cache.textContent = "not loaded";
  } else {
    els.cache.textContent = downloaded
      ? "downloaded and cached"
      : "from browser cache";
  }
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

function handleProgress(message) {
  const filePart =
    message.total > 0 ? Math.min(1, message.loaded / message.total) : 0;
  const overall = Math.round(
    ((message.index - 1 + filePart) / message.count) * 100,
  );
  updateProgress(overall);
  const file = String(message.file ?? "");
  if (file.startsWith("GPU ")) {
    els.loadStatus.textContent = `Uploading weights to the GPU (${overall}%).`;
    return;
  }
  if (!message.cached) {
    downloaded = true;
  }
  const size = message.total
    ? `${formatBytes(message.loaded)} of ${formatBytes(message.total)}`
    : formatBytes(message.loaded);
  const action = message.cached ? "Reading from cache" : "Downloading";
  els.loadStatus.textContent = `${action} ${file} (${overall}%, ${size}).`;
}

function handlePageWorkerError(message) {
  modelReady = null;
  if (pageClient) {
    pageClient.terminate();
    pageClient = null;
  }
  if (loading) {
    return;
  }
  loading = false;
  busy = false;
  generating = false;
  els.loadPanel.hidden = false;
  els.loadButton.hidden = false;
  els.progressWrap.hidden = true;
  els.loadStatus.textContent = `The worker stopped: ${message}`;
  els.demoStatus.textContent = "The model is not loaded.";
  els.benchStatus.textContent = "Load the model above first.";
  els.backendNote.hidden = true;
  els.running.textContent = "not loaded";
  els.cache.textContent = "not loaded";
  setRunningState();
  updateControls();
}

async function loadModel(options = {}) {
  if (loading || modelReady !== null) {
    return;
  }
  loading = true;
  downloaded = false;
  updateProgress(0);
  els.loadPanel.hidden = false;
  els.running.textContent = "starting";
  els.cache.textContent = options.switching ? "cached" : "not loaded";
  els.loadStatus.textContent = options.switching
    ? "Switching engine. Weights stay in the browser's cache."
    : "Starting the worker.";
  els.demoStatus.textContent = "Loading the model.";
  els.backendNote.hidden = true;
  updateControls();

  try {
    if (!pageClient) {
      pageClient = new EngineClient(requestedBackend(), {
        onProgress: handleProgress,
        onError: handlePageWorkerError,
      });
    }
    const message = await pageClient.load(WEIGHTS);
    loading = false;
    modelReady = message;
    els.loadPanel.hidden = true;
    els.progressWrap.hidden = true;
    setRunningState();

    const fallback =
      message.requestedBackend === "webgpu" && message.backend !== "webgpu";
    let status;
    if (fallback) {
      const reason = message.fallback
        ? ` (${message.fallback})`
        : "";
      status = `Model ready on the CPU engine.`;
      els.backendNote.textContent = `WebGPU is not available here${reason}. The CPU engine is running.`;
      els.backendNote.hidden = false;
    } else {
      status = downloaded
        ? "Model ready, downloaded and cached."
        : "Model ready, from browser cache.";
      els.backendNote.hidden = true;
    }
    status += ` Loaded in ${formatLoadTime(message.loadMs)}.`;
    els.demoStatus.textContent = status;
    if (deviceBench && benchResults.length === 0) {
      els.benchStatus.textContent = "Ready to run.";
    }
    updateControls();
  } catch (error) {
    loading = false;
    modelReady = null;
    if (pageClient) {
      pageClient.terminate();
      pageClient = null;
    }
    els.loadPanel.hidden = false;
    els.loadButton.hidden = false;
    els.progressWrap.hidden = true;
    els.loadStatus.textContent = `Could not load the model: ${describeError(error)}`;
    els.demoStatus.textContent = "The model is not loaded.";
    els.benchStatus.textContent = "Load the model above first.";
    els.backendNote.hidden = true;
    els.running.textContent = "not loaded";
    els.cache.textContent = "not loaded";
    updateControls();
  }
}

function handleBackendChange() {
  const mode = els.backendMode.value;
  if (mode === backendMode) {
    return;
  }
  backendMode = mode;
  if (loading || busy || modelReady === null || !pageClient) {
    updateControls();
    return;
  }
  const required = requestedBackend();
  if (pageClient.requested === required) {
    return;
  }
  pageClient.terminate();
  pageClient = null;
  modelReady = null;
  busy = false;
  generating = false;
  els.running.textContent = "starting";
  els.cache.textContent = "cached";
  els.backendNote.hidden = true;
  loadModel({ switching: true });
}

function stopGeneration() {
  if (!generating) {
    return;
  }
  stopRequested = true;
  if (pageClient) {
    pageClient.terminate();
    pageClient = null;
  }
  modelReady = null;
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
  els.backendNote.hidden = true;
  els.running.textContent = "not loaded";
  els.cache.textContent = "in browser cache";
  updateControls();
}

async function generate() {
  if (modelReady === null || busy || !pageClient) {
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
    const encoded = await pageClient.request({ type: "encode", text });
    if (!encoded.ids || encoded.ids.length === 0) {
      throw new Error("The prompt produced no tokens.");
    }
    els.demoStatus.textContent = "Prefilling the prompt.";
    const startedAt = performance.now();
    const done = await pageClient.request(
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
          els.meterSpeed.textContent = formatRate((tokenCount - 1) / seconds);
        }
      }
    }
    const speed = els.meterSpeed.textContent;
    els.demoStatus.textContent = `Generated ${formatCount(tokenCount)} tokens${
      speed !== "\u2013" ? ` at ${speed} tokens/s` : ""
    } on ${describeEngine(modelReady)}.`;
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

function runMetrics(promptTokens, done, fallbackCount) {
  const timings = Array.isArray(done.timings) ? done.timings : [];
  const tokens = Array.isArray(done.tokens) ? done.tokens : [];
  if (timings.length === 0) {
    throw new Error("The worker returned no timings.");
  }
  const tokenCount = tokens.length || fallbackCount;
  const ttft = Number(timings[0]);
  const decodeSeconds =
    timings.length > 1
      ? (Number(timings[timings.length - 1]) - ttft) / 1000
      : 0;
  return {
    ttft,
    tokens: tokenCount,
    decode: decodeSeconds > 0 ? (tokenCount - 1) / decodeSeconds : 0,
    prefill: ttft > 0 ? promptTokens / (ttft / 1000) : 0,
  };
}

function median(values) {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) {
    return sorted[middle];
  }
  return (sorted[middle - 1] + sorted[middle]) / 2;
}

function medianMetrics(runs) {
  return {
    decode: median(runs.map((run) => run.decode)),
    prefill: median(runs.map((run) => run.prefill)),
    ttft: median(runs.map((run) => run.ttft)),
    tokens: runs[0]?.tokens ?? 0,
  };
}

function benchProgress(message, label) {
  const filePart =
    message.total > 0 ? Math.min(1, message.loaded / message.total) : 0;
  const overall = Math.round(
    ((message.index - 1 + filePart) / message.count) * 100,
  );
  if (String(message.file ?? "").startsWith("GPU ")) {
    els.benchStatus.textContent = `${label}: uploading weights to the GPU (${overall}%).`;
  } else {
    els.benchStatus.textContent = `${label}: preparing the engine (${overall}%).`;
  }
}

async function benchmarkBackend(requested, label) {
  const client = new EngineClient(requested, {
    onProgress: (message) => benchProgress(message, label),
  });
  try {
    const ready = await client.load(WEIGHTS);
    if (requested === "webgpu" && ready.backend !== "webgpu") {
      return { ready, skipped: true, label };
    }
    const encoded = await client.request({
      type: "encode",
      text: deviceBench.prompt,
    });
    if (!encoded.ids || encoded.ids.length === 0) {
      throw new Error("The benchmark prompt produced no tokens.");
    }
    const warmups = Math.max(0, Number(deviceBench.warmup) || 0);
    for (let index = 0; index < warmups; index += 1) {
      els.benchStatus.textContent = `${label}: warm-up ${index + 1} of ${warmups}.`;
      await client.request({
        type: "generate",
        ids: encoded.ids,
        maxNewTokens: Number(deviceBench.max_new_tokens),
        stream: false,
      });
    }
    const runs = Math.max(1, Number(deviceBench.runs) || 1);
    const measured = [];
    for (let index = 0; index < runs; index += 1) {
      els.benchStatus.textContent = `${label}: timed run ${index + 1} of ${runs}.`;
      const done = await client.request({
        type: "generate",
        ids: encoded.ids,
        maxNewTokens: Number(deviceBench.max_new_tokens),
        stream: false,
      });
      measured.push(
        runMetrics(
          encoded.ids.length,
          done,
          Number(deviceBench.max_new_tokens),
        ),
      );
    }
    return { ready, label, metrics: medianMetrics(measured) };
  } finally {
    client.terminate();
  }
}

function projectEntryFor(ready) {
  const web = payloadData?.web ?? [];
  if (web.length === 0) {
    return null;
  }
  const candidates = web.filter((entry) => isQuantized(entry));
  if (ready.backend === "webgpu") {
    return (
      candidates.find((entry) => entry.milestone === "W4") ??
      candidates.find((entry) => entry.target === "webgpu") ??
      null
    );
  }
  if ((Number(ready.threads) || 1) > 1) {
    return (
      candidates.find((entry) => entry.milestone === "W3") ??
      candidates.find((entry) => entry.target === "wasm-threads") ??
      null
    );
  }
  if (ready.simd) {
    return (
      candidates.find((entry) => entry.milestone === "W2") ??
      candidates.find((entry) => entry.target === "wasm-simd") ??
      null
    );
  }
  return (
    candidates.find((entry) => entry.milestone === "W1") ??
    candidates.find((entry) => entry.target === "wasm") ??
    null
  );
}

function appendMetricCells(row, metrics) {
  const cells = [
    ["Decode tok/s", formatRate(metrics.decode)],
    ["Prompt tok/s", formatRate(metrics.prefill)],
    ["First token", formatMs(metrics.ttft)],
  ];
  for (const [label, text] of cells) {
    const cell = document.createElement("td");
    cell.dataset.label = label;
    cell.textContent = text;
    row.append(cell);
  }
}

function renderBenchResults() {
  els.benchBody.replaceChildren();
  let missingProject = false;

  for (const result of benchResults) {
    const project = projectEntryFor(result.ready);
    if (project === null) {
      missingProject = true;
    }
    const rowspan = project === null ? 1 : 2;

    const deviceRow = document.createElement("tr");
    const engineCell = document.createElement("th");
    engineCell.scope = "row";
    engineCell.rowSpan = rowspan;
    const engineName = document.createElement("strong");
    engineName.textContent = result.label;
    const engineDetail = document.createElement("span");
    engineDetail.className = "engine-detail";
    engineDetail.textContent = describeEngine(result.ready);
    engineCell.append(engineName, engineDetail);

    const deviceSource = document.createElement("td");
    deviceSource.className = "source-cell";
    deviceSource.dataset.label = "Source";
    deviceSource.textContent = "this device";

    deviceRow.append(engineCell, deviceSource);
    appendMetricCells(deviceRow, result.metrics);
    els.benchBody.append(deviceRow);

    if (project !== null) {
      const projectRow = document.createElement("tr");
      projectRow.className = "project-row";
      const projectSource = document.createElement("td");
      projectSource.className = "source-cell";
      projectSource.dataset.label = "Source";
      projectSource.textContent = `${project.milestone} \u00b7 ${
        project.browser ?? "browser"
      }`;
      projectRow.append(projectSource);
      appendMetricCells(projectRow, {
        decode: project.decode_tok_s,
        prefill: project.prefill_tok_s,
        ttft: project.ttft_ms,
      });
      els.benchBody.append(projectRow);
    }
  }

  els.benchTableWrap.hidden = benchResults.length === 0;
  if (missingProject && benchResults.length > 0) {
    els.benchNote.textContent =
      "The project has no recorded browser run for this engine yet.";
    els.benchNote.hidden = false;
  }
}

function suggestBackend() {
  const cpu = benchResults.find((result) => result.ready.backend === "cpu");
  const gpu = benchResults.find((result) => result.ready.backend === "webgpu");
  if (!cpu || !gpu) {
    return;
  }
  if (cpu.metrics.decode > gpu.metrics.decode) {
    els.benchHint.textContent = `The CPU engine decoded faster on this device: ${formatRate(
      cpu.metrics.decode,
    )} vs ${formatRate(gpu.metrics.decode)} tokens/s. Use the engine control above to switch to CPU.`;
    els.benchHint.hidden = false;
  }
}

async function runBenchmark() {
  if (modelReady === null || busy || !deviceBench || benchRunning) {
    return;
  }
  busy = true;
  benchRunning = true;
  benchResults = [];
  let benchFailure = null;
  let benchUnavailable = null;
  let benchError = null;
  els.benchBody.replaceChildren();
  els.benchTableWrap.hidden = true;
  els.benchNote.hidden = true;
  els.benchHint.hidden = true;
  els.benchStatus.textContent = "Running the CPU benchmark.";
  updateControls();

  try {
    const cpu = await benchmarkBackend("cpu", "CPU");
    benchResults.push(cpu);
    renderBenchResults();
  } catch (error) {
    benchFailure = describeError(error);
  }

  if (benchFailure === null) {
    try {
      const gpu = await benchmarkBackend("webgpu", "WebGPU");
      if (gpu.skipped) {
        benchUnavailable =
          gpu.ready.fallback || "WebGPU is not available on this device";
      } else {
        benchResults.push(gpu);
        renderBenchResults();
      }
    } catch (error) {
      benchError = describeError(error);
    }
  }

  busy = false;
  benchRunning = false;
  suggestBackend();
  if (benchResults.length > 0) {
    els.benchTableWrap.hidden = false;
  }

  if (benchFailure) {
    els.benchStatus.textContent = `Benchmark failed: ${benchFailure}`;
  } else if (benchUnavailable) {
    els.benchStatus.textContent = `Finished on CPU. WebGPU is not available here: ${benchUnavailable}`;
    els.benchNote.textContent = `The WebGPU engine was requested but ran the CPU engine instead: ${benchUnavailable}.`;
    els.benchNote.hidden = false;
  } else if (benchError) {
    els.benchStatus.textContent = `Finished on CPU. WebGPU failed: ${benchError}`;
    els.benchNote.textContent = `The WebGPU run failed, so only the CPU result is shown.`;
    els.benchNote.hidden = false;
  } else {
    const warmups = Math.max(0, Number(deviceBench.warmup) || 0);
    const runs = Math.max(1, Number(deviceBench.runs) || 1);
    els.benchStatus.textContent = `Finished: ${formatCount(
      warmups,
    )} warm-up${warmups === 1 ? "" : "s"} and ${formatCount(
      runs,
    )} timed run${runs === 1 ? "" : "s"} on each engine.`;
  }
  updateControls();
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

function nativeCorrectness(entry, protocol) {
  const total = Number(protocol.tokens_checked);
  if (entry.milestone === "R0-torch") {
    return "The reference run. Every engine is checked against these tokens.";
  }
  if (isQuantized(entry)) {
    const parts = [];
    const rate = Number(entry.token_match_rate);
    if (Number.isFinite(rate)) {
      const matched = Number.isFinite(total) ? Math.round(rate * total) : null;
      const matchedText =
        matched === null
          ? ""
          : ` (${formatCount(matched)} of ${formatCount(total)})`;
      parts.push(
        `${Math.round(rate * 100)}% of reference tokens match${matchedText}`,
      );
    }
    const perplexity = Number(entry.perplexity);
    const perplexityF32 = Number(entry.perplexity_f32);
    if (Number.isFinite(perplexity)) {
      let text = `perplexity ${perplexity.toFixed(2)}`;
      if (Number.isFinite(perplexityF32)) {
        const delta = (perplexity / perplexityF32 - 1) * 100;
        text += ` vs ${perplexityF32.toFixed(2)} f32 (${delta >= 0 ? "+" : ""}${delta.toFixed(1)}%)`;
      }
      parts.push(text);
    }
    return `int8: ${parts.join("; ")}.`;
  }
  if (entry.matches_reference) {
    return `Matches the reference on all ${formatCount(total)} tokens.`;
  }
  const rate = Number(entry.token_match_rate);
  if (Number.isFinite(rate)) {
    const matched = Number.isFinite(total) ? Math.round(rate * total) : null;
    const matchedText =
      matched === null
        ? ""
        : ` (${formatCount(matched)} of ${formatCount(total)})`;
    return `${Math.round(rate * 100)}% of tokens match${matchedText}.`;
  }
  return "This milestone does not match the reference exactly.";
}

function webCorrectness(entry, protocol) {
  const total = Number(protocol.tokens_checked);
  const parts = [];
  if (entry.matches_native) {
    parts.push("produced exactly the native int8 engine's tokens");
  }
  if (entry.matches_reference) {
    parts.push(`matches the reference on all ${formatCount(total)} tokens`);
  } else {
    const rate = Number(entry.token_match_rate);
    if (Number.isFinite(rate)) {
      const matched = Number.isFinite(total) ? Math.round(rate * total) : null;
      const matchedText =
        matched === null
          ? ""
          : ` (${formatCount(matched)} of ${formatCount(total)})`;
      parts.push(
        `${Math.round(rate * 100)}% of reference tokens match${matchedText}`,
      );
    }
  }
  if (parts.length === 0) {
    return "No correctness result is recorded for this run.";
  }
  const text = parts.join("; ");
  return `${text.charAt(0).toUpperCase()}${text.slice(1)}.`;
}

function appendCompareNote(article, entry, baseline) {
  if (
    !entry.baseline ||
    !baseline ||
    String(baseline.milestone) !== String(entry.baseline)
  ) {
    return;
  }
  const note = document.createElement("p");
  note.className = "compare-note";
  note.textContent = `Compared with ${entry.baseline}.`;
  article.append(note);
}

function buildThreadTable(entry, scaling, protocol) {
  const rows = scaling.map((item) => ({
    threads: Number(item.threads ?? String(item.milestone).replace(/\D/g, "")),
    item,
  }));
  const cores = Number(protocol.cores);
  if (
    Number.isFinite(cores) &&
    cores > 0 &&
    !rows.some((row) => row.threads === cores)
  ) {
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
  for (const label of ["Threads", "Decode tok/s", "Prompt tok/s"]) {
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

function renderNativeCards(entries, protocol) {
  els.nativeMilestones.replaceChildren();
  const byMilestone = new Map(
    entries.map((entry) => [String(entry.milestone), entry]),
  );
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

    for (const part of [entry.date, entry.commit]) {
      if (part) {
        const node = document.createElement("span");
        node.textContent = part;
        meta.append(node);
      }
    }
    if (
      entry.sessions !== undefined &&
      Number(entry.sessions) !== Number(protocol.sessions)
    ) {
      const node = document.createElement("span");
      node.textContent = `${formatCount(entry.sessions)} session${
        Number(entry.sessions) === 1 ? "" : "s"
      }`;
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

    const baseline =
      (entry.baseline && byMilestone.get(String(entry.baseline))) ||
      previous;
    const showBefore =
      baseline !== null && !isReference(entry) && entry.milestone !== "N1";

    const deltas = document.createElement("ul");
    deltas.className = "deltas";
    appendDelta(
      deltas,
      "Decode",
      showBefore ? baseline.decode_tok_s : null,
      entry.decode_tok_s,
      "tok/s",
      formatRate,
    );
    appendDelta(
      deltas,
      "Prefill",
      showBefore ? baseline.prefill_tok_s : null,
      entry.prefill_tok_s,
      "tok/s",
      formatRate,
    );
    appendDelta(
      deltas,
      "First token",
      showBefore ? baseline.ttft_ms : null,
      entry.ttft_ms,
      "",
      formatMs,
    );
    appendDelta(
      deltas,
      "Peak memory",
      showBefore ? baseline.peak_mb : null,
      entry.peak_mb,
      "",
      formatMemory,
    );

    const correctness = document.createElement("p");
    correctness.className = "correctness";
    correctness.textContent = nativeCorrectness(entry, protocol);

    article.append(meta, title, headline, deltas);
    appendCompareNote(article, entry, baseline);
    article.append(correctness);

    if (Array.isArray(entry.scaling) && entry.scaling.length > 0) {
      article.append(buildThreadTable(entry, entry.scaling, protocol));
    }

    if (entry.journal) {
      const link = document.createElement("a");
      link.className = "journal-link";
      link.href = REPO_URL + entry.journal;
      link.textContent = "Journal note";
      article.append(link);
    }

    els.nativeMilestones.append(article);
    previous = entry;
  }
}

function tokenMatchText(row) {
  const matched = Number(row.matched_tokens);
  const total = Number(row.tokens_total);
  if (!Number.isFinite(matched) || !Number.isFinite(total)) {
    return "\u2013";
  }
  return `${formatCount(matched)} / ${formatCount(total)}`;
}

function renderRace(tables) {
  els.raceTables.replaceChildren();
  for (const tableData of tables) {
    const card = document.createElement("div");
    card.className = "race-card";

    const heading = document.createElement("h4");
    heading.textContent = tableData.label;

    const note = document.createElement("p");
    note.className = "block-note";
    note.textContent = tableData.note;

    const scroll = document.createElement("div");
    scroll.className = "table-scroll";

    const table = document.createElement("table");
    table.className = "data-table race-table";
    const caption = document.createElement("caption");
    caption.className = "sr-only";
    caption.textContent = `${tableData.label}: engines in the same container.`;
    table.append(caption);

    const head = document.createElement("thead");
    const headRow = document.createElement("tr");
    for (const label of [
      "Engine",
      "Decode tok/s",
      "Prompt tok/s",
      "First token",
      "Tokens matching",
    ]) {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = label;
      headRow.append(cell);
    }
    head.append(headRow);
    table.append(head);

    const body = document.createElement("tbody");
    for (const row of tableData.rows) {
      const tr = document.createElement("tr");
      if (row.quantized) {
        tr.className = "row--int8";
      }
      const engineCell = document.createElement("th");
      engineCell.scope = "row";
      const label = document.createElement("span");
      label.className = "row-label";
      label.textContent = row.label;
      const runId = document.createElement("span");
      runId.className = "run-id";
      runId.textContent = row.milestone;
      engineCell.append(label, runId);
      tr.append(engineCell);
      const cells = [
        ["Decode tok/s", formatRate(row.decode_tok_s)],
        ["Prompt tok/s", formatRate(row.prefill_tok_s)],
        ["First token", formatMs(row.ttft_ms)],
        ["Tokens matching", tokenMatchText(row)],
      ];
      for (const [cellLabel, text] of cells) {
        const cell = document.createElement("td");
        cell.dataset.label = cellLabel;
        cell.textContent = text;
        tr.append(cell);
      }
      body.append(tr);
    }
    table.append(body);
    scroll.append(table);
    card.append(heading, note, scroll);
    els.raceTables.append(card);
  }
}

function sizeRowLabel(row) {
  const label = document.createElement("span");
  label.className = "row-label";
  label.textContent = row.label;
  const runId = document.createElement("span");
  runId.className = "run-id";
  runId.textContent = row.milestone;
  return [label, runId];
}

function renderSizes(groups, protocol) {
  els.sizesBody.replaceChildren();
  const memory = Number(protocol.memory_gb);
  const oomText = Number.isFinite(memory)
    ? `did not fit in ${formatCount(memory)} GB`
    : "did not fit";
  for (const [groupIndex, group] of groups.entries()) {
    for (const [rowIndex, row] of group.rows.entries()) {
      const tr = document.createElement("tr");
      tr.dataset.model = group.label;
      if (rowIndex === 0 && groupIndex > 0) {
        tr.className = "group-start";
      }
      if (rowIndex === 0) {
        const modelCell = document.createElement("th");
        modelCell.scope = "rowgroup";
        modelCell.rowSpan = group.rows.length;
        modelCell.className = "model-cell";
        modelCell.textContent = group.label;
        tr.append(modelCell);
      }
      const engineCell = document.createElement("th");
      engineCell.scope = "row";
      engineCell.append(...sizeRowLabel(row));
      tr.append(engineCell);

      if (row.status === "oom") {
        const cell = document.createElement("td");
        cell.colSpan = 4;
        cell.className = "oom";
        cell.dataset.label = "Result";
        cell.textContent = oomText;
        tr.append(cell);
      } else {
        const cells = [
          ["Decode tok/s", formatRate(row.decode_tok_s)],
          ["Prompt tok/s", formatRate(row.prefill_tok_s)],
          ["Peak memory", formatMemory(row.peak_mb)],
          ["Perplexity", formatPerplexity(row.perplexity)],
        ];
        for (const [label, text] of cells) {
          const cell = document.createElement("td");
          cell.dataset.label = label;
          cell.textContent = text;
          tr.append(cell);
        }
      }
      els.sizesBody.append(tr);
    }
  }
}

function variantMatch(variant) {
  if (variant.matches_native) {
    return "native int8";
  }
  if (variant.matches_reference) {
    return "reference";
  }
  const rate = Number(variant.token_match_rate);
  if (Number.isFinite(rate)) {
    return `${Math.round(rate * 100)}%`;
  }
  return "\u2013";
}

function buildVariantTable(entry) {
  if (!Array.isArray(entry.variants) || entry.variants.length === 0) {
    return null;
  }
  const scroll = document.createElement("div");
  scroll.className = "table-scroll";
  const table = document.createElement("table");
  table.className = "data-table variant-table";
  const caption = document.createElement("caption");
  caption.className = "sr-only";
  caption.textContent = `Other runs of ${entry.milestone}.`;
  table.append(caption);

  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const label of [
    "Run",
    "Browser",
    "Weights",
    "Threads",
    "Decode tok/s",
    "Prompt tok/s",
    "First token",
    "Match",
  ]) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = label;
    headRow.append(cell);
  }
  head.append(headRow);
  table.append(head);

  const body = document.createElement("tbody");
  for (const variant of entry.variants) {
    const tr = document.createElement("tr");
    const runCell = document.createElement("th");
    runCell.scope = "row";
    const runLabel = document.createElement("span");
    runLabel.className = "row-label";
    runLabel.textContent = variant.milestone;
    const target = document.createElement("span");
    target.className = "run-id";
    target.textContent = variant.target ?? "";
    runCell.append(runLabel, target);
    tr.append(runCell);

    const cells = [
      ["Browser", variant.browser ?? "\u2013"],
      ["Weights", variant.weights ?? "\u2013"],
      [
        "Threads",
        variant.threads === undefined ? "\u2013" : formatCount(variant.threads),
      ],
      ["Decode tok/s", formatRate(variant.decode_tok_s)],
      ["Prompt tok/s", formatRate(variant.prefill_tok_s)],
      ["First token", formatMs(variant.ttft_ms)],
      ["Match", variantMatch(variant)],
    ];
    for (const [label, text] of cells) {
      const cell = document.createElement("td");
      cell.dataset.label = label;
      cell.textContent = text;
      tr.append(cell);
    }
    body.append(tr);
  }
  table.append(body);
  scroll.append(table);
  return scroll;
}

function renderWebCards(entries, protocol) {
  els.webMilestones.replaceChildren();
  const byMilestone = new Map(
    entries.map((entry) => [String(entry.milestone), entry]),
  );
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
    for (const part of [entry.browser, entry.target, entry.date]) {
      if (part) {
        const node = document.createElement("span");
        node.textContent = part;
        meta.append(node);
      }
    }
    if (entry.threads !== undefined) {
      const node = document.createElement("span");
      const count = Number(entry.threads);
      node.textContent = `${formatCount(entry.threads)} thread${
        count === 1 ? "" : "s"
      }`;
      meta.append(node);
    }
    const weights = document.createElement("span");
    weights.className = "tag";
    weights.textContent = entry.weights ?? "weights";
    meta.append(weights);
    if (
      entry.sessions !== undefined &&
      Number(entry.sessions) !== Number(protocol.sessions)
    ) {
      const node = document.createElement("span");
      node.textContent = `${formatCount(entry.sessions)} session${
        Number(entry.sessions) === 1 ? "" : "s"
      }`;
      meta.append(node);
    }

    const title = document.createElement("h3");
    title.textContent = entry.title;

    const headline = document.createElement("p");
    headline.className = "milestone-headline";
    headline.textContent = entry.headline;

    const baseline =
      (entry.baseline && byMilestone.get(String(entry.baseline))) || previous;
    const showBefore = baseline !== null;

    const deltas = document.createElement("ul");
    deltas.className = "deltas";
    appendDelta(
      deltas,
      "Decode",
      showBefore ? baseline.decode_tok_s : null,
      entry.decode_tok_s,
      "tok/s",
      formatRate,
    );
    appendDelta(
      deltas,
      "Prompt",
      showBefore ? baseline.prefill_tok_s : null,
      entry.prefill_tok_s,
      "tok/s",
      formatRate,
    );
    appendDelta(
      deltas,
      "First token",
      showBefore ? baseline.ttft_ms : null,
      entry.ttft_ms,
      "",
      formatMs,
    );

    const correctness = document.createElement("p");
    correctness.className = "correctness";
    correctness.textContent = webCorrectness(entry, protocol);

    article.append(meta, title, headline, deltas);
    appendCompareNote(article, entry, baseline);
    article.append(correctness);

    const variants = buildVariantTable(entry);
    if (variants) {
      article.append(variants);
    }

    if (entry.journal) {
      const link = document.createElement("a");
      link.className = "journal-link";
      link.href = REPO_URL + entry.journal;
      link.textContent = "Journal note";
      article.append(link);
    }

    els.webMilestones.append(article);
    previous = entry;
  }
}

function renderChart(container, entries, options = {}) {
  const valid = entries.filter(
    (entry) => Number.isFinite(Number(entry.decode_tok_s)) && Number(entry.decode_tok_s) > 0,
  );
  if (valid.length === 0) {
    container.replaceChildren();
    chartWidths.set(container, 0);
    return;
  }

  const compact = options.compact === true;
  const width = Math.max(240, Math.floor(container.clientWidth || 600));
  chartWidths.set(container, width);
  const narrow = width < 430;
  const longestLabel = valid.reduce(
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
  const rowHeight = compact ? 28 : narrow ? 30 : 32;
  const barHeight = compact ? 13 : narrow ? 14 : 16;
  const top = 20;
  const axisHeight = 28;
  const left = labelWidth;
  const right = width - valueWidth;
  const plotWidth = Math.max(80, right - left);
  const height = top + valid.length * rowHeight + axisHeight;

  const values = valid.map((entry) => Number(entry.decode_tok_s));
  const minLog = Math.floor(Math.log10(Math.min(...values)));
  const maxLog = Math.ceil(Math.log10(Math.max(...values)) * 10) / 10;
  const logSpan = Math.max(0.1, maxLog - minLog);
  const xFor = (value) =>
    left + ((Math.log10(value) - minLog) / logSpan) * plotWidth;
  const hatchId = `${container.id || "chart"}-hatch`;

  const parts = [
    `<svg class="chart-svg${
      narrow ? " chart-svg--narrow" : ""
    }" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" aria-hidden="true" focusable="false">`,
    `<defs><pattern id="${hatchId}" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect class="chart-hatch-bg" width="6" height="6"/><line class="chart-hatch-line" x1="0" y1="0" x2="0" y2="6"/></pattern></defs>`,
  ];

  for (let exponent = minLog; exponent <= Math.floor(maxLog); exponent += 1) {
    const value = 10 ** exponent;
    const x = xFor(value);
    parts.push(
      `<line class="chart-grid" x1="${x.toFixed(1)}" y1="${
        top - 6
      }" x2="${x.toFixed(1)}" y2="${top + valid.length * rowHeight}"/>`,
    );
    const anchor = x < left + 12 ? "start" : x > right - 12 ? "end" : "middle";
    parts.push(
      `<text class="chart-tick" x="${x.toFixed(1)}" y="${
        top + valid.length * rowHeight + 16
      }" text-anchor="${anchor}">${value}</text>`,
    );
  }

  valid.forEach((entry, index) => {
    const value = Number(entry.decode_tok_s);
    const y = top + index * rowHeight + (rowHeight - barHeight) / 2;
    const rowCenter = y + barHeight / 2 + 4;
    const quantized = isQuantized(entry);
    const comparison = isReference(entry);
    const barWidth = Math.max(2, xFor(value) - left);
    const barClass = quantized
      ? "chart-bar-int8"
      : comparison
        ? "chart-bar-ref"
        : "chart-bar-ours";
    const fill =
      quantized
        ? ` fill="url(#${hatchId})"`
        : "";
    parts.push(
      `<text class="chart-label" x="2" y="${rowCenter.toFixed(
        1,
      )}">${escapeXml(entry.milestone)}</text>`,
    );
    parts.push(
      `<rect class="${barClass}"${fill} x="${left}" y="${y.toFixed(
        1,
      )}" width="${barWidth.toFixed(1)}" height="${barHeight}" rx="2"/>`,
    );
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
  container.innerHTML = parts.join("");
}

function scheduleChart(container) {
  const entries = chartData.get(container) ?? [];
  if (entries.length === 0) {
    return;
  }
  const width = Math.max(240, Math.floor(container.clientWidth || 600));
  if (width !== chartWidths.get(container)) {
    renderChart(container, entries, { compact: container === els.webChart });
  }
}

function describeSessionNote(native, protocol) {
  const n1 = native.find((entry) => entry.milestone === "N1");
  if (
    n1 &&
    Number.isFinite(Number(n1.sessions)) &&
    Number(n1.sessions) !== Number(protocol.sessions)
  ) {
    return `N1 used ${formatCount(n1.sessions)} session${
      Number(n1.sessions) === 1 ? "" : "s"
    } and no warm-up. From N2 on, each milestone uses ${formatCount(
      protocol.sessions,
    )} sessions.`;
  }
  return "";
}

function renderStory(payload) {
  const protocol = payload.protocol ?? {};
  const native = payload.native ?? [];
  const race = payload.race ?? [];
  const sizes = payload.sizes ?? [];
  const web = payload.web ?? [];
  const hasStory =
    native.length > 0 || race.length > 0 || sizes.length > 0 || web.length > 0;

  els.story.hidden = !hasStory;
  if (!hasStory) {
    return;
  }

  els.storyNative.hidden = native.length === 0;
  els.storyRace.hidden = race.length === 0;
  els.storySizes.hidden = sizes.length === 0;
  els.storyWeb.hidden = web.length === 0;

  if (native.length > 0) {
    chartData.set(els.nativeChart, native);
    renderChart(els.nativeChart, native);
    renderNativeCards(native, protocol);
  }
  if (race.length > 0) {
    renderRace(race);
  }
  if (sizes.length > 0) {
    renderSizes(sizes, protocol);
  }
  if (web.length > 0) {
    chartData.set(els.webChart, web);
    renderChart(els.webChart, web, { compact: true });
    renderWebCards(web, protocol);
  }

  const note = describeSessionNote(native, protocol);
  els.storyNote.textContent = note;
  els.storyNote.hidden = note.length === 0;
}

function renderDevice(payload) {
  const protocol = payload.protocol ?? {};
  const bench = payload.benchmark ?? null;
  deviceBench = bench;

  if (!bench) {
    els.deviceCopy.textContent =
      "The benchmark configuration could not be loaded.";
    els.benchStatus.textContent = "The benchmark is unavailable.";
    return;
  }

  const warmups = Number(bench.warmup);
  const runs = Number(bench.runs);
  els.deviceCopy.textContent = `Run prompt ${bench.prompt_id} with ${formatCount(
    bench.max_new_tokens,
  )} new tokens: ${formatCount(warmups)} warm-up${
    warmups === 1 ? "" : "s"
  }, then ${formatCount(runs)} timed run${
    runs === 1 ? "" : "s"
  }. It runs on each engine this device supports, one after the other. The project rows are browser measurements from this repository on the development laptop, not the container.`;
  els.benchStatus.textContent =
    modelReady === null ? "Load the model above first." : "Ready to run.";
}

function renderMethod(protocol) {
  for (const element of document.querySelectorAll("[data-protocol]")) {
    element.textContent = formatProtocol(
      element.dataset.protocol,
      protocol[element.dataset.protocol],
    );
  }
}

function renderData(payload) {
  payloadData = payload;
  const protocol = payload.protocol ?? {};
  els.weights.textContent = weightsLabel();
  updateLoadButton();
  renderMethod(protocol);
  renderDevice(payload);
  renderStory(payload);
  if (modelReady !== null) {
    setRunningState();
  }
  if (modelReady !== null && benchResults.length === 0) {
    els.benchStatus.textContent = "Ready to run.";
  }
  updateControls();
}

function renderDataError(error) {
  console.error(error);
  payloadData = null;
  deviceBench = null;
  els.story.hidden = true;
  els.storyIntro.textContent = "The milestone data could not be loaded.";
  const message = document.createElement("p");
  message.className = "status";
  message.textContent = "The milestone data is unavailable.";
  els.nativeMilestones.replaceChildren(message);
  els.nativeChart.replaceChildren();
  els.webChart.replaceChildren();
  els.storyNote.textContent = "";
  els.deviceCopy.textContent = "The benchmark configuration could not be loaded.";
  els.benchStatus.textContent = "The benchmark is unavailable.";
  els.weights.textContent = "int8";
  for (const element of document.querySelectorAll("[data-protocol]")) {
    element.textContent = "\u2013";
  }
  updateControls();
}

function dataUrl() {
  const raw = new URLSearchParams(location.search).get("data");
  if (!raw) {
    return DEFAULT_DATA_URL;
  }
  if (
    /^[a-z][a-z0-9+.-]*:/i.test(raw) ||
    raw.startsWith("//") ||
    raw.startsWith("/")
  ) {
    return DEFAULT_DATA_URL;
  }
  try {
    const url = new URL(raw, location.href);
    return url.origin === location.origin ? url.href : DEFAULT_DATA_URL;
  } catch {
    return DEFAULT_DATA_URL;
  }
}

async function fetchData() {
  const response = await fetch(dataUrl(), { cache: "no-cache" });
  if (!response.ok) {
    throw new Error(`data file: HTTP ${response.status}`);
  }
  return response.json();
}

function linkRepo() {
  for (const link of document.querySelectorAll("[data-repo]")) {
    link.href = REPO_URL + link.dataset.repo;
  }
  for (const link of document.querySelectorAll("[data-tree]")) {
    link.href = REPO_TREE_URL + link.dataset.tree;
  }
}

function wireEvents() {
  els.loadButton.addEventListener("click", () => loadModel());
  els.generate.addEventListener("click", generate);
  els.stop.addEventListener("click", stopGeneration);
  els.runBench.addEventListener("click", runBenchmark);
  els.backendMode.addEventListener("change", handleBackendChange);
  window.addEventListener("resize", () => {
    for (const container of [els.nativeChart, els.webChart]) {
      scheduleChart(container);
    }
  });
}

function observeCharts() {
  if (!("ResizeObserver" in window)) {
    return;
  }
  const observer = new ResizeObserver(() => {
    for (const container of [els.nativeChart, els.webChart]) {
      scheduleChart(container);
    }
  });
  observer.observe(els.nativeChart);
  observer.observe(els.webChart);
}

async function init() {
  wireEvents();
  linkRepo();
  observeCharts();

  const initial = new URLSearchParams(location.search).get("backend");
  if (initial === "cpu" || initial === "webgpu") {
    backendMode = initial;
    els.backendMode.value = initial;
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
