const PKG = "pkg-threads";

let started = false;

self.onmessage = (event) => {
  const message = event.data;
  if (started || message.type !== "init") {
    return;
  }
  started = true;
  start(message).catch((error) => {
    self.postMessage({ type: "error", message: describe(error) });
  });
};

async function start({ module, memory, pointer, index }) {
  const binding = await import(`./${PKG}/gpt2_web.js`);
  binding.initSync({ module, memory });
  self.postMessage({ type: "ready", index });
  binding.thread_pool_worker_entry(pointer, index);
}

function describe(error) {
  return error instanceof Error ? error.message : String(error);
}
