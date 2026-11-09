import init, { Engine } from "./pkg/gpt2_web.js";

const MODEL_BASE_URL = "/models/gpt2/";
const CACHE_NAME = "gpt2-weights-v1";
const TEXT_DECODER = new TextDecoder();

let engine = null;

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
          values: engine.logits(Uint32Array.from(message.ids)),
        });
        break;
      case "generate":
        generate(message);
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

async function load(weights) {
  const started = performance.now();
  const weightsFile =
    weights === "f32" ? "model.safetensors" : "model-int8.safetensors";
  const names = ["config.json", "vocab.json", "merges.txt", weightsFile];

  const files = {};
  for (let index = 0; index < names.length; index += 1) {
    files[names[index]] = await readFile(names[index], index, names.length);
  }

  await init();
  const buildStart = performance.now();
  engine = new Engine(
    TEXT_DECODER.decode(files["config.json"]),
    TEXT_DECODER.decode(files["vocab.json"]),
    TEXT_DECODER.decode(files["merges.txt"]),
    files[weightsFile],
    weights,
  );
  self.postMessage({
    type: "ready",
    weights,
    loadMs: performance.now() - started,
    buildMs: performance.now() - buildStart,
  });
}

async function readFile(name, index, count) {
  const url = new URL(name, new URL(MODEL_BASE_URL, self.location.origin)).href;
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

  if ("caches" in self) {
    const cache = await caches.open(CACHE_NAME);
    const cached = await cache.match(url);
    if (cached) {
      const bytes = new Uint8Array(await cached.arrayBuffer());
      progress(bytes.byteLength, bytes.byteLength, true);
      return bytes;
    }
    const bytes = await download(url, progress);
    await cache.put(
      url,
      new Response(bytes, {
        headers: {
          "Content-Type": "application/octet-stream",
          "Content-Length": String(bytes.byteLength),
        },
      }),
    );
    return bytes;
  }

  return readFileFromOpfs(url, name, progress);
}

async function readFileFromOpfs(url, name, progress) {
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

  const bytes = await download(url, progress);
  const handle = await directory.getFileHandle(fileName, { create: true });
  const writable = await handle.createWritable();
  await writable.write(bytes);
  await writable.close();
  return bytes;
}

async function download(url, progress) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url}: HTTP ${response.status} ${response.statusText}`);
  }
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body) {
    const buffer = await response.arrayBuffer();
    progress(buffer.byteLength, total || buffer.byteLength);
    return new Uint8Array(buffer);
  }

  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  let reported = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    chunks.push(value);
    loaded += value.byteLength;
    const now = performance.now();
    if (now - reported > 100) {
      progress(loaded, total);
      reported = now;
    }
  }
  progress(loaded, total || loaded);

  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

function generate(message) {
  const started = performance.now();
  const tokens = [];
  const timings = [];
  engine.generate(
    Uint32Array.from(message.ids),
    message.maxNewTokens,
    (token) => {
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
    },
  );
  self.postMessage({
    type: "done",
    id: message.id,
    timings,
    tokens,
  });
}

function describe(error) {
  return error instanceof Error ? error.message : String(error);
}
