const MODEL_BASE_URL = "/models/gpt2/";
const CACHE_NAME = "gpt2-weights-v2";
const CHUNK_BYTES = 64 * 1024 * 1024;
const PART_PARALLEL = 3;
const MANIFEST_TTL_MS = 5 * 60 * 1000;
const TEXT_DECODER = new TextDecoder();
const PARAMS = new URL(self.location.href).searchParams;
const MODEL_BASE = parseModelBase(PARAMS.get("model"));
const REQUESTED_PKG = ["pkg", "pkg-nosimd", "pkg-threads"].includes(
  PARAMS.get("pkg"),
)
  ? PARAMS.get("pkg")
  : "pkg";
const REQUESTED_THREADS = parseThreads(PARAMS.get("threads"));
const REQUESTED_BACKEND = PARAMS.get("backend") === "webgpu" ? "webgpu" : "cpu";
const GPU_PIPELINE_DEPTH = parseDepth(PARAMS.get("pipeline"));

let Engine = null;
let binding = null;
let engine = null;
let enginePkg = null;
let wasmModule = null;
let threads = 1;
let poolWorkers = [];
let gpu = null;
let backend = "cpu";
let fallback = null;
let gpuQueue = Promise.resolve();
let nCtx = 1024;

self.onmessage = async (event) => {
  const message = event.data;
  try {
    switch (message.type) {
      case "load":
        await load(message.weights);
        break;
      case "encode":
        self.postMessage({
          type: "encoded",
          id: message.id,
          ids: engine.encode(message.text),
        });
        break;
      case "decode":
        self.postMessage({
          type: "decoded",
          id: message.id,
          text: engine.decode(Uint32Array.from(message.ids)),
        });
        break;
      case "logits":
        self.postMessage({
          type: "logits",
          id: message.id,
          values: await enqueue(() => computeLogits(message.ids)),
        });
        break;
      case "generate":
        await enqueue(() => generate(message));
        break;
      default:
        throw new Error(`unknown worker message ${message.type}`);
    }
  } catch (error) {
    self.postMessage({
      type: "error",
      id: message.id,
      message: describe(error),
    });
  }
};

function enqueue(task) {
  const result = gpuQueue.then(task, task);
  gpuQueue = result.catch(() => {});
  return result;
}

async function computeLogits(ids) {
  if (gpu) {
    return gpu.logits(Uint32Array.from(ids));
  }
  return engine.logits(Uint32Array.from(ids));
}

async function load(weights) {
  const started = performance.now();
  const weightsFile =
    weights === "f32" ? "model.safetensors" : "model-int8.safetensors";
  const names = ["config.json", "vocab.json", "merges.txt"];
  const count = names.length + 1;

  const files = {};
  for (let index = 0; index < names.length; index += 1) {
    files[names[index]] = await readFile(names[index], index, count);
  }
  const manifest = await readManifest();
  const entry = manifest?.weights;
  files[weightsFile] =
    entry && entry.name === weightsFile
      ? await readParts(entry, names.length, count)
      : await readFile(weightsFile, names.length, count);

  await importEngine();
  const buildStart = performance.now();
  const config = JSON.parse(TEXT_DECODER.decode(files["config.json"]));
  nCtx = Number(config.n_ctx) || nCtx;
  const gpuRequested = REQUESTED_BACKEND === "webgpu";
  const cpuThreads = threads;
  engine = new Engine(
    TEXT_DECODER.decode(files["config.json"]),
    TEXT_DECODER.decode(files["vocab.json"]),
    TEXT_DECODER.decode(files["merges.txt"]),
    files[weightsFile],
    weights,
    gpuRequested ? 1 : threads,
  );
  let gpuInfo = null;
  if (gpuRequested) {
    try {
      gpuInfo = await startGpu(config, weights);
      engine.release_weights();
      backend = "webgpu";
    } catch (error) {
      fallback = describe(error);
      gpu = null;
      backend = "cpu";
      if (cpuThreads !== 1) {
        threads = cpuThreads;
        engine = new Engine(
          TEXT_DECODER.decode(files["config.json"]),
          TEXT_DECODER.decode(files["vocab.json"]),
          TEXT_DECODER.decode(files["merges.txt"]),
          files[weightsFile],
          weights,
          threads,
        );
      }
    }
  }
  if (gpu === null && threads > 1) {
    await startPool();
  }
  self.postMessage({
    type: "ready",
    weights,
    backend,
    nCtx,
    requestedBackend: REQUESTED_BACKEND,
    fallback,
    pkg: backend === "webgpu" ? "webgpu" : enginePkg,
    simd: binding.simd_enabled(),
    threads: backend === "webgpu" ? 1 : threads,
    pipelineDepth: gpu ? GPU_PIPELINE_DEPTH : 0,
    gpu: gpuInfo,
    loadMs: performance.now() - started,
    buildMs: performance.now() - buildStart,
  });
}

async function startGpu(config, weights) {
  const { GpuEngine, requestGpu } = await import("./gpu/engine.js");
  const { device, info } = await requestGpu();
  gpu = new GpuEngine({
    device,
    source: engine,
    config,
    weights,
    pipelineDepth: GPU_PIPELINE_DEPTH,
  });
  await gpu.init((name, index, count) => {
    self.postMessage({
      type: "progress",
      file: `GPU ${name}`,
      index,
      count,
      loaded: index,
      total: count,
      cached: false,
    });
  });
  return {
    adapter: info,
    limits: {
      max_buffer_size: device.limits.maxBufferSize,
      max_storage_buffer_binding_size:
        device.limits.maxStorageBufferBindingSize,
      max_compute_workgroup_storage_size:
        device.limits.maxComputeWorkgroupStorageSize,
      max_storage_buffers_per_shader_stage:
        device.limits.maxStorageBuffersPerShaderStage,
    },
    memory: gpu.memory(),
    pipeline_depth: gpu.depth,
  };
}

async function importEngine() {
  if (binding !== null) {
    return;
  }
  const candidates =
    REQUESTED_PKG === "pkg-threads" ? ["pkg-threads", "pkg"] : [REQUESTED_PKG];
  let lastError = null;
  for (const candidate of candidates) {
    try {
      const imported = await import(`./${candidate}/gpt2_web.js`);
      if (candidate === "pkg-threads") {
        if (!self.crossOriginIsolated) {
          throw new Error("WebAssembly threads need cross-origin isolation");
        }
        const url = new URL(`./${candidate}/gpt2_web_bg.wasm`, import.meta.url);
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`${url}: HTTP ${response.status}`);
        }
        wasmModule = await WebAssembly.compile(await response.arrayBuffer());
        imported.initSync({ module: wasmModule });
        threads = REQUESTED_THREADS;
      } else {
        await imported.default();
        wasmModule = null;
        threads = 1;
      }
      binding = imported;
      Engine = imported.Engine;
      enginePkg = candidate;
      return;
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError;
}

async function startPool() {
  const workers = binding.thread_pool_workers();
  if (workers !== threads - 1) {
    throw new Error(
      `the engine asked for ${threads - 1} workers, the build offers ${workers}`,
    );
  }
  const memory = binding.shared_memory();
  const pointer = binding.thread_pool_pointer();
  const ready = [];
  for (let index = 0; index < workers; index += 1) {
    const worker = new Worker(new URL("./thread.js", import.meta.url), {
      type: "module",
    });
    poolWorkers.push(worker);
    ready.push(
      new Promise((resolve, reject) => {
        worker.onmessage = (event) => {
          const message = event.data;
          if (message.type === "ready") {
            resolve();
          } else if (message.type === "error") {
            reject(new Error(message.message));
          }
        };
        worker.onerror = (event) =>
          reject(new Error(event.message || "The thread worker failed."));
        worker.postMessage({
          type: "init",
          module: wasmModule,
          memory,
          pointer,
          index,
        });
      }),
    );
  }
  try {
    await Promise.all(ready);
  } catch (error) {
    for (const worker of poolWorkers) {
      worker.terminate();
    }
    poolWorkers = [];
    throw error;
  }
}

function parseThreads(value) {
  if (value === null || value === "") {
    return Math.min(navigator.hardwareConcurrency || 1, 8);
  }
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed < 1) {
    return 1;
  }
  return Math.min(parsed, 32);
}

async function readFile(name, index, count) {
  const url = new URL(name, MODEL_BASE).href;
  const progress = (loaded, total, cached) =>
    self.postMessage({
      type: "progress",
      file: name,
      index: index + 1,
      count,
      loaded,
      total,
      cached,
    });
  const report = throttled((loaded, total) =>
    progress(loaded, total, false),
  );

  if ("caches" in self) {
    const cache = await caches.open(CACHE_NAME);
    const bytes = await readCached(cache, url);
    if (bytes) {
      progress(bytes.byteLength, bytes.byteLength, true);
      return bytes;
    }
    const downloaded = await download(url, report);
    report(downloaded.byteLength, downloaded.byteLength, true);
    await writeCached(cache, url, downloaded);
    return downloaded;
  }

  return readFileFromOpfs(url, name, progress, report);
}

async function readManifest() {
  const url = new URL("manifest.json", MODEL_BASE).href;
  const cache = "caches" in self ? await caches.open(CACHE_NAME) : null;
  const cached = cache ? await readCachedManifest(cache, url) : null;
  if (cached && Date.now() - cached.fetched <= MANIFEST_TTL_MS) {
    return cached.manifest;
  }

  let response;
  try {
    response = await fetch(url);
  } catch (error) {
    if (cached) {
      return cached.manifest;
    }
    throw new Error(`could not read ${url}: ${describe(error)}`);
  }
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    if (cached) {
      return cached.manifest;
    }
    throw new Error(`${url}: HTTP ${response.status} ${response.statusText}`);
  }

  let manifest;
  try {
    manifest = await response.json();
  } catch {
    throw new Error(`${url} is not valid JSON`);
  }
  if (!isManifest(manifest)) {
    throw new Error(`${url} has no usable weights entry`);
  }
  await cacheManifest(cache, url, manifest);
  return manifest;
}

async function readCachedManifest(cache, url) {
  const response = await cache.match(url);
  if (!response) {
    return null;
  }
  try {
    const record = await response.json();
    if (!record || !Number.isFinite(record.fetched)) {
      return null;
    }
    if (!isManifest(record.manifest)) {
      return null;
    }
    return { fetched: record.fetched, manifest: record.manifest };
  } catch {
    return null;
  }
}

function isManifest(manifest) {
  const entry = manifest?.weights;
  if (!entry || typeof entry.name !== "string") {
    return false;
  }
  if (!Number.isInteger(entry.bytes) || entry.bytes < 1) {
    return false;
  }
  if (typeof entry.sha256 !== "string") {
    return false;
  }
  if (!Array.isArray(entry.parts) || entry.parts.length === 0) {
    return false;
  }
  return entry.parts.every(
    (part) =>
      part &&
      typeof part.name === "string" &&
      Number.isInteger(part.bytes) &&
      part.bytes > 0 &&
      typeof part.sha256 === "string",
  );
}

async function cacheManifest(cache, url, manifest) {
  if (!cache) {
    return;
  }
  try {
    await cache.put(
      url,
      new Response(JSON.stringify({ fetched: Date.now(), manifest }), {
        headers: { "Content-Type": "application/json" },
      }),
    );
  } catch (error) {
    // A full or unavailable cache must not stop the model from running.
  }
}

async function readParts(entry, index, count) {
  const parts = entry.parts;
  const total = parts.reduce((sum, part) => sum + part.bytes, 0);
  if (total !== entry.bytes) {
    throw new Error(
      `model manifest: parts hold ${total} bytes, expected ${entry.bytes}`,
    );
  }

  const bytes = new Uint8Array(entry.bytes);
  const received = new Uint32Array(parts.length);
  const offsets = new Uint32Array(parts.length);
  let offset = 0;
  for (let part = 0; part < parts.length; part += 1) {
    offsets[part] = offset;
    offset += parts[part].bytes;
  }

  const cache = "caches" in self ? await caches.open(CACHE_NAME) : null;
  const missing = [];
  if (cache) {
    for (let partIndex = 0; partIndex < parts.length; partIndex += 1) {
      const part = parts[partIndex];
      const url = new URL(part.name, MODEL_BASE).href;
      const response = await cache.match(url);
      if (!response) {
        missing.push(partIndex);
        continue;
      }
      const chunk = new Uint8Array(await response.arrayBuffer());
      if (chunk.byteLength !== part.bytes) {
        await cache.delete(url);
        missing.push(partIndex);
        continue;
      }
      bytes.set(chunk, offsets[partIndex]);
      received[partIndex] = part.bytes;
    }
  } else {
    for (let part = 0; part < parts.length; part += 1) {
      missing.push(part);
    }
  }

  const progress = (loaded, cached) =>
    self.postMessage({
      type: "progress",
      file: entry.name,
      index: index + 1,
      count,
      loaded,
      total: entry.bytes,
      cached,
    });
  const loadedBytes = () => received.reduce((sum, value) => sum + value, 0);

  if (missing.length === 0) {
    progress(entry.bytes, true);
    return bytes;
  }

  const report = throttled(() => progress(loadedBytes(), false));
  report(0, 0, true);

  const downloadPart = async (partIndex) => {
    const part = parts[partIndex];
    const url = new URL(part.name, MODEL_BASE).href;
    const chunk = await download(url, (loaded) => {
      received[partIndex] = loaded;
      report();
    });
    if (chunk.byteLength !== part.bytes) {
      throw new Error(
        `${part.name}: expected ${part.bytes} bytes, got ${chunk.byteLength}`,
      );
    }
    const digest = await sha256Hex(chunk);
    if (digest !== part.sha256) {
      throw new Error(
        `${part.name}: sha256 mismatch (expected ${part.sha256}, got ${digest})`,
      );
    }
    bytes.set(chunk, offsets[partIndex]);
    received[partIndex] = part.bytes;
    if (cache) {
      try {
        await cache.put(
          url,
          new Response(chunk, {
            headers: {
              "Content-Type": "application/octet-stream",
              "Content-Length": String(chunk.byteLength),
            },
          }),
        );
      } catch (error) {
        // A full or unavailable cache must not stop the model from running.
      }
    }
    report(0, 0, true);
  };

  await forEachLimited(missing, PART_PARALLEL, downloadPart);
  report(0, 0, true);
  return bytes;
}

function parseModelBase(value) {
  let url;
  try {
    url = new URL(value || MODEL_BASE_URL, self.location.origin);
  } catch {
    throw new Error(`invalid model base URL: ${value}`);
  }
  if (!url.pathname.endsWith("/")) {
    url.pathname += "/";
  }
  return url.href;
}

function throttled(report) {
  let last = 0;
  return (loaded, total, force = false) => {
    const now = performance.now();
    if (!force && now - last < 100) {
      return;
    }
    last = now;
    report(loaded, total);
  };
}

async function forEachLimited(items, limit, task) {
  let next = 0;
  const runners = Array.from(
    { length: Math.min(limit, items.length) },
    async () => {
      for (;;) {
        const current = next;
        next += 1;
        if (current >= items.length) {
          return;
        }
        await task(items[current]);
      }
    },
  );
  await Promise.all(runners);
}

async function sha256Hex(bytes) {
  if (!crypto?.subtle) {
    throw new Error(
      "sha256 verification needs a secure context (https or localhost)",
    );
  }
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  let hex = "";
  for (const byte of digest) {
    hex += byte.toString(16).padStart(2, "0");
  }
  return hex;
}

function partUrl(url, part) {
  const key = new URL(url);
  key.search = `part=${part}`;
  return key.href;
}

async function readCached(cache, url) {
  const manifestResponse = await cache.match(partUrl(url, "manifest"));
  if (!manifestResponse) {
    return null;
  }
  const manifest = await manifestResponse.json();
  const bytes = new Uint8Array(manifest.bytes);
  let offset = 0;
  for (let part = 0; part < manifest.parts; part += 1) {
    const response = await cache.match(partUrl(url, part));
    if (!response) {
      return null;
    }
    const chunk = new Uint8Array(await response.arrayBuffer());
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return offset === bytes.byteLength ? bytes : null;
}

async function writeCached(cache, url, bytes) {
  const parts = Math.max(1, Math.ceil(bytes.byteLength / CHUNK_BYTES));
  for (let part = 0; part < parts; part += 1) {
    const start = part * CHUNK_BYTES;
    const chunk = bytes.subarray(
      start,
      Math.min(start + CHUNK_BYTES, bytes.byteLength),
    );
    await cache.put(
      partUrl(url, part),
      new Response(chunk, {
        headers: {
          "Content-Type": "application/octet-stream",
          "Content-Length": String(chunk.byteLength),
        },
      }),
    );
  }
  await cache.put(
    partUrl(url, "manifest"),
    new Response(JSON.stringify({ parts, bytes: bytes.byteLength }), {
      headers: { "Content-Type": "application/json" },
    }),
  );
}

async function readFileFromOpfs(url, name, progress, report) {
  const root = await navigator.storage.getDirectory();
  const directory = await root.getDirectoryHandle("gpt2-weights", {
    create: true,
  });
  const fileName = name.replaceAll("/", "_");
  try {
    const handle = await directory.getFileHandle(fileName);
    const file = await handle.getFile();
    const bytes = new Uint8Array(await file.arrayBuffer());
    progress(bytes.byteLength, bytes.byteLength, true);
    return bytes;
  } catch (error) {
    if (error.name !== "NotFoundError") {
      throw error;
    }
  }

  const bytes = await download(url, report);
  report(bytes.byteLength, bytes.byteLength, true);
  const handle = await directory.getFileHandle(fileName, { create: true });
  const writable = await handle.createWritable();
  await writable.write(bytes);
  await writable.close();
  return bytes;
}

async function download(url, report) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url}: HTTP ${response.status} ${response.statusText}`);
  }
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body) {
    const buffer = await response.arrayBuffer();
    report?.(buffer.byteLength, total || buffer.byteLength);
    return new Uint8Array(buffer);
  }

  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    chunks.push(value);
    loaded += value.byteLength;
    report?.(loaded, total || loaded);
  }

  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

async function generate(message) {
  const started = performance.now();
  const tokens = [];
  const timings = [];
  const sampling = message.sampling ?? null;
  const endToken = Number.isInteger(message.endToken) ? message.endToken : null;
  let reason = "max tokens";
  const ids = Uint32Array.from(message.ids);
  if (ids.length === 0) {
    throw new Error("generate requires at least one token id");
  }
  if (ids.length >= nCtx) {
    throw new Error(
      `the prompt uses ${ids.length} tokens, more than the ${nCtx}-token context`,
    );
  }
  const maxNewTokens = Math.max(
    0,
    Math.min(
      Math.floor(Number(message.maxNewTokens) || 0),
      nCtx - ids.length,
    ),
  );
  const onToken = (token) => {
    if (endToken !== null && token === endToken) {
      reason = "end of text";
      return false;
    }
    timings.push(performance.now() - started);
    tokens.push(token);
    if (message.stream) {
      self.postMessage({
        type: "token",
        id: message.id,
        token,
        text: engine.decode(Uint32Array.from(tokens)),
      });
    }
    return true;
  };
  if (gpu) {
    await gpu.generate(ids, maxNewTokens, onToken, sampling);
  } else if (sampling) {
    const ended = engine.generate_sampled(
      ids,
      maxNewTokens,
      sampling.temperature,
      Math.max(1, Math.round(sampling.topK)),
      sampling.topP,
      sampling.repetitionPenalty,
      sampling.seed >>> 0,
      endToken ?? 0xffffffff,
      (token) => {
        onToken(token);
      },
    );
    if (ended) {
      reason = "end of text";
    }
  } else if (endToken === null) {
    engine.generate(ids, maxNewTokens, onToken);
  } else {
    const ended = engine.generate_until(ids, maxNewTokens, endToken, (token) => {
      onToken(token);
    });
    if (ended) {
      reason = "end of text";
    }
  }
  self.postMessage({
    type: "done",
    id: message.id,
    timings,
    tokens,
    reason,
    seed: sampling ? sampling.seed >>> 0 : null,
  });
}

function parseDepth(value) {
  const parsed = Number.parseInt(value ?? "", 10);
  if (!Number.isFinite(parsed) || parsed < 1) {
    return 4;
  }
  return Math.min(parsed, 16);
}

function describe(error) {
  return error instanceof Error ? error.message : String(error);
}
