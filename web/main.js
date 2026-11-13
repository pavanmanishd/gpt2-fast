const WEIGHTS = "int8";
const MAX_NEW_TOKENS = 128;

const worker = new Worker(new URL("./worker.js", import.meta.url), {
  type: "module",
});

const statusLine = document.getElementById("status");
const promptInput = document.getElementById("prompt");
const generateButton = document.getElementById("generate");
const output = document.getElementById("output");
const ttftValue = document.getElementById("ttft");
const speedValue = document.getElementById("speed");
const tokenValue = document.getElementById("tokens");

const pending = new Map();
let nextId = 1;
let ready = false;

worker.onmessage = (event) => {
  const message = event.data;
  if (message.type === "progress") {
    showProgress(message);
    return;
  }
  if (message.type === "ready") {
    ready = true;
    generateButton.disabled = false;
    statusLine.textContent = `Ready in ${(message.loadMs / 1000).toFixed(1)} s (${message.weights} weights, ${(message.buildMs / 1000).toFixed(1)} s to build).`;
    return;
  }
  if (message.type === "error" && message.id === undefined) {
    statusLine.textContent = `Failed: ${message.message}`;
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
  statusLine.textContent = `Worker failed: ${event.message}`;
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
  const size = message.total
    ? `${formatBytes(message.loaded)} / ${formatBytes(message.total)}`
    : formatBytes(message.loaded);
  const source = message.cached ? "from browser storage" : size;
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

async function generate() {
  if (!ready) {
    return;
  }
  const text = promptInput.value;
  if (text.length === 0) {
    return;
  }

  generateButton.disabled = true;
  output.textContent = "";
  tokenValue.textContent = "0";
  speedValue.textContent = "\u2013";
  ttftValue.textContent = "\u2013";

  let started = 0;
  let firstTokenAt = 0;
  let tokens = 0;

  try {
    const encoded = await request({ type: "encode", text });
    started = performance.now();
    await request(
      {
        type: "generate",
        ids: encoded.ids,
        maxNewTokens: MAX_NEW_TOKENS,
        stream: true,
        endToken: 50256,
      },
      (message) => {
        const now = performance.now();
        if (tokens === 0) {
          firstTokenAt = now;
          ttftValue.textContent = `${(now - started).toFixed(0)} ms`;
        }
        tokens += 1;
        tokenValue.textContent = String(tokens);
        if (tokens > 1) {
          const seconds = (now - firstTokenAt) / 1000;
          if (seconds > 0) {
            speedValue.textContent = ((tokens - 1) / seconds).toFixed(1);
          }
        }
        output.textContent = message.text;
      },
    );
    statusLine.textContent = `Generated ${tokens} tokens at ${speedValue.textContent} tokens/s.`;
  } catch (error) {
    statusLine.textContent = `Generation failed: ${error.message}`;
  } finally {
    generateButton.disabled = !ready;
  }
}

generateButton.addEventListener("click", generate);
worker.postMessage({ type: "load", weights: WEIGHTS });
