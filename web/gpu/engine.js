import * as shaders from "./shaders.js";

const COMPUTE = globalThis.GPUShaderStage?.COMPUTE ?? 0;

function uniform(binding) {
  return { binding, visibility: COMPUTE, buffer: { type: "uniform" } };
}

function readOnly(binding) {
  return { binding, visibility: COMPUTE, buffer: { type: "read-only-storage" } };
}

function readWrite(binding) {
  return { binding, visibility: COMPUTE, buffer: { type: "storage" } };
}

function entry(binding, buffer) {
  return { binding, resource: { buffer } };
}

function align(value, multiple) {
  return Math.ceil(value / multiple) * multiple;
}

function writeUniform(device, buffer, values) {
  const bytes = new ArrayBuffer(16);
  const view = new DataView(bytes);
  for (const [offset, type, value] of values) {
    if (type === "u32") {
      view.setUint32(offset, value, true);
    } else {
      view.setFloat32(offset, value, true);
    }
  }
  device.queue.writeBuffer(buffer, 0, bytes);
}

export async function requestGpu() {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not available");
  }
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) {
    throw new Error("WebGPU returned no adapter");
  }
  if (adapter.isFallbackAdapter === true) {
    throw new Error("WebGPU offered a software fallback adapter");
  }
  const info = {};
  if (adapter.info) {
    for (const key of ["vendor", "architecture", "device", "description"]) {
      info[key] = adapter.info[key] ?? "";
    }
  }
  const requiredLimits = {};
  for (const key of [
    "maxBufferSize",
    "maxStorageBufferBindingSize",
    "maxComputeWorkgroupStorageSize",
    "maxStorageBuffersPerShaderStage",
  ]) {
    if (adapter.limits[key] !== undefined) {
      requiredLimits[key] = adapter.limits[key];
    }
  }
  let device;
  try {
    device = await adapter.requestDevice({ requiredLimits });
  } catch {
    device = await adapter.requestDevice();
  }
  return { device, info };
}

export class GpuEngine {
  constructor({ device, source, config, weights, pipelineDepth }) {
    this.device = device;
    this.source = source;
    this.config = config;
    this.int8 = weights === "int8";
    this.bytes = {
      weights: 0,
      activations: 0,
      kv_cache: 0,
      staging: 0,
    };
    this.depth = Math.max(1, pipelineDepth);
    this.linearWorkgroup = this.int8 ? 32 : 64;
  }

  async init(onProgress) {
    const { n_embd, n_head } = this.config;
    const head_dim = n_embd / n_head;
    if (!Number.isInteger(head_dim) || head_dim > 64) {
      throw new Error(`unsupported head dimension ${head_dim}`);
    }
    if (n_embd % 64 !== 0) {
      throw new Error(`unsupported embedding size ${n_embd}`);
    }
    const limits = this.device.limits;
    if (limits.maxComputeWorkgroupStorageSize < 8192) {
      throw new Error("WebGPU workgroup memory limit is too small");
    }
    if (limits.maxComputeInvocationsPerWorkgroup < 256) {
      throw new Error("WebGPU workgroup size limit is too small");
    }
    if (limits.maxStorageBuffersPerShaderStage < 6) {
      throw new Error("WebGPU storage buffer binding limit is too small");
    }
    if (limits.maxStorageBufferBindingSize < 16 * 1024 * 1024) {
      throw new Error("WebGPU storage buffer limit is too small");
    }
    if (limits.maxBufferSize < 64 * 1024 * 1024) {
      throw new Error("WebGPU buffer limit is too small");
    }
    if (this.config.vocab_size > limits.maxComputeWorkgroupsPerDimension) {
      throw new Error("WebGPU dispatch limit is too small for the vocabulary");
    }

    this.manifest = JSON.parse(this.source.weight_manifest());
    this.layouts = this.createLayouts();
    this.pipelines = await this.createPipelines();
    await this.uploadWeights(onProgress);
    this.createActivations();
    this.createPasses();
  }

  createLayouts() {
    const device = this.device;
    const layout = (entries) => device.createBindGroupLayout({ entries });
    return {
      step: layout([uniform(0), readWrite(1)]),
      embedF32: layout([
        readOnly(0),
        readOnly(1),
        readOnly(2),
        readWrite(3),
        uniform(4),
      ]),
      embedInt8: layout([
        readOnly(0),
        readOnly(1),
        readOnly(2),
        readOnly(3),
        readOnly(4),
        readWrite(5),
        uniform(6),
      ]),
      linearF32: layout([
        readOnly(0),
        readOnly(1),
        readOnly(2),
        readWrite(3),
        uniform(4),
      ]),
      linearInt8: layout([
        readOnly(0),
        readOnly(1),
        readOnly(2),
        readOnly(3),
        readWrite(4),
        uniform(5),
      ]),
      layerNorm: layout([
        readOnly(0),
        readOnly(1),
        readOnly(2),
        readWrite(3),
        uniform(4),
      ]),
      kv: layout([readOnly(0), readWrite(1), readWrite(2), uniform(3)]),
      attention: layout([
        readOnly(0),
        readOnly(1),
        readOnly(2),
        readWrite(3),
        uniform(4),
      ]),
      residual: layout([readWrite(0), readOnly(1), uniform(2)]),
      headF32: layout([
        readOnly(0),
        readOnly(1),
        readOnly(2),
        readWrite(3),
        uniform(4),
      ]),
      headInt8: layout([
        readOnly(0),
        readOnly(1),
        readOnly(2),
        readOnly(3),
        readOnly(4),
        readWrite(5),
        uniform(6),
      ]),
      argmax: layout([readOnly(0), uniform(1)]),
      sampleReset: layout([readWrite(0)]),
      samplePenalty: layout([
        readWrite(0),
        readOnly(1),
        readWrite(2),
        readWrite(3),
        uniform(4),
      ]),
      sampleBucket: layout([readOnly(0), readWrite(1), uniform(2)]),
      sampleGather: layout([
        readOnly(0),
        readOnly(1),
        readWrite(2),
        readWrite(3),
        uniform(4),
      ]),
      sampleToken: layout([
        readOnly(0),
        readOnly(1),
        readWrite(2),
        readWrite(3),
        uniform(4),
      ]),
    };
  }

  async createPipelines() {
    const device = this.device;
    const layouts = this.layouts;
    const build = async (code, bindGroupLayouts) => {
      const module = device.createShaderModule({ code });
      const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts,
      });
      return device.createComputePipelineAsync({
        layout: pipelineLayout,
        compute: { module, entryPoint: "main" },
      });
    };
    const step = layouts.step;
    return {
      embed: await build(
        this.int8 ? shaders.embedInt8 : shaders.embedF32,
        [step, this.int8 ? layouts.embedInt8 : layouts.embedF32],
      ),
      linear: await build(
        this.int8 ? shaders.linearInt8 : shaders.linearF32,
        [step, this.int8 ? layouts.linearInt8 : layouts.linearF32],
      ),
      linearDecode: await build(
        this.int8 ? shaders.linearDecodeInt8 : shaders.linearDecodeF32,
        [step, this.int8 ? layouts.linearInt8 : layouts.linearF32],
      ),
      layerNorm: await build(shaders.layerNorm, [step, layouts.layerNorm]),
      kv: await build(shaders.kvStore, [step, layouts.kv]),
      attention: await build(shaders.attention, [step, layouts.attention]),
      residual: await build(shaders.residual, [step, layouts.residual]),
      head: await build(
        this.int8 ? shaders.headDecodeInt8 : shaders.headDecodeF32,
        [step, this.int8 ? layouts.headInt8 : layouts.headF32],
      ),
      argmax: await build(shaders.argmax, [step, layouts.argmax]),
      sampleReset: await build(shaders.sampleReset, [
        step,
        layouts.sampleReset,
      ]),
      samplePenalty: await build(shaders.samplePenalty, [
        step,
        layouts.samplePenalty,
      ]),
      sampleBucket: await build(shaders.sampleBucket, [
        step,
        layouts.sampleBucket,
      ]),
      sampleGather: await build(shaders.sampleGather, [
        step,
        layouts.sampleGather,
      ]),
      sampleToken: await build(shaders.sampleToken, [
        step,
        layouts.sampleToken,
      ]),
    };
  }

  createBuffer(
    size,
    usage = GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  ) {
    const padded = Math.max(16, align(size, 16));
    return this.device.createBuffer({ size: padded, usage });
  }

  upload(data) {
    const bytes =
      data instanceof Uint8Array
        ? data
        : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const size = Math.max(16, align(bytes.byteLength, 16));
    const buffer = this.device.createBuffer({
      size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    const view = new Uint8Array(buffer.getMappedRange());
    view.set(bytes.subarray(0, Math.min(bytes.byteLength, size)));
    buffer.unmap();
    this.bytes.weights += size;
    return buffer;
  }

  uploadChunk(data, count, offset) {
    const view = data.subarray(offset, offset + count);
    return this.upload(view);
  }

  async uploadWeights(onProgress) {
    this.tensors = new Map();
    this.dummy = this.createBuffer(16);
    const maxBinding = Math.min(
      this.device.limits.maxStorageBufferBindingSize,
      this.device.limits.maxBufferSize,
    );
    const total = this.manifest.length;
    for (let index = 0; index < total; index += 1) {
      const tensor = this.manifest[index];
      this.tensors.set(tensor.name, this.uploadTensor(tensor, maxBinding));
      if (onProgress) {
        onProgress(tensor.name, index + 1, total);
      }
    }
  }

  uploadTensor(tensor, maxBinding) {
    const [rows, columns] = tensor.shape;
    const int8 = tensor.int8 && this.int8;
    if (tensor.int8 && !this.int8) {
      throw new Error(`tensor ${tensor.name} is quantized but weights are f32`);
    }
    if (tensor.shape.length === 1) {
      const data = this.source.weight(tensor.name);
      if (!data) {
        throw new Error(`missing tensor ${tensor.name}`);
      }
      return { buffers: [this.upload(data)], int8: false };
    }
    if (columns % 4 !== 0) {
      throw new Error(`tensor ${tensor.name} has an unsupported width`);
    }
    const groups = columns / 64;
    if (int8 && columns % 64 !== 0) {
      throw new Error(`tensor ${tensor.name} is not a multiple of the int8 group size`);
    }
    if (int8) {
      const data = this.source.weight_int8(tensor.name);
      const scales = this.source.weight_scales(tensor.name);
      if (!data || !scales) {
        throw new Error(`missing int8 tensor ${tensor.name}`);
      }
      const wordsPerRow = columns / 4;
      const rowBytes = columns;
      const rowsPerBuffer = Math.floor(maxBinding / Math.max(16, rowBytes));
      if (rowsPerBuffer < 1) {
        throw new Error("storage buffer limit cannot hold one weight row");
      }
      const chunks = Math.ceil(rows / rowsPerBuffer);
      if (chunks > 2) {
        throw new Error("storage buffer limit needs more than two weight buffers");
      }
      const buffers = [];
      for (let chunk = 0; chunk < chunks; chunk += 1) {
        const first = chunk * rowsPerBuffer;
        const count = Math.min(rowsPerBuffer, rows - first);
        buffers.push(
          this.uploadChunk(data, count * wordsPerRow, first * wordsPerRow),
        );
      }
      const scaleBuffers = [];
      for (let chunk = 0; chunk < chunks; chunk += 1) {
        const first = chunk * rowsPerBuffer;
        const count = Math.min(rowsPerBuffer, rows - first);
        scaleBuffers.push(
          this.uploadChunk(scales, count * groups, first * groups),
        );
      }
      return {
        buffers,
        scales: scaleBuffers,
        rowsPerBuffer,
        int8: true,
        n_out: rows,
        n_in: columns,
      };
    }

    const data = this.source.weight(tensor.name);
    if (!data) {
      throw new Error(`missing tensor ${tensor.name}`);
    }
    const rowBytes = columns * 4;
    const rowsPerBuffer = Math.floor(maxBinding / Math.max(16, rowBytes));
    if (tensor.name === "wte.weight") {
      if (rowsPerBuffer < 1) {
        throw new Error("storage buffer limit cannot hold one weight row");
      }
      const chunks = Math.ceil(rows / rowsPerBuffer);
      if (chunks > 2) {
        throw new Error("storage buffer limit needs more than two weight buffers");
      }
      const buffers = [];
      for (let chunk = 0; chunk < chunks; chunk += 1) {
        const first = chunk * rowsPerBuffer;
        const count = Math.min(rowsPerBuffer, rows - first);
        buffers.push(this.uploadChunk(data, count * columns, first * columns));
      }
      return { buffers, rowsPerBuffer, int8: false, n_out: rows, n_in: columns };
    }
    if (rows * rowBytes > maxBinding) {
      throw new Error(`tensor ${tensor.name} does not fit in one storage buffer`);
    }
    return {
      buffers: [this.upload(data)],
      int8: false,
      n_out: rows,
      n_in: columns,
    };
  }

  tensor(name) {
    const tensor = this.tensors.get(name);
    if (!tensor) {
      throw new Error(`missing tensor ${name}`);
    }
    return tensor;
  }

  createActivations() {
    const { n_embd, n_ctx, vocab_size } = this.config;
    const scalar = (count) => this.createBuffer(count * 4);
    this.x = scalar(n_ctx * n_embd);
    this.norm = scalar(n_ctx * n_embd);
    this.attn = scalar(n_ctx * n_embd);
    this.projection = scalar(n_ctx * n_embd);
    this.qkv = scalar(n_ctx * 3 * n_embd);
    this.hidden = scalar(n_ctx * 4 * n_embd);
    this.logitsBuffer = scalar(vocab_size);
    this.tokens = scalar(n_ctx);
    this.bytes.activations = [
      this.x,
      this.norm,
      this.attn,
      this.projection,
      this.qkv,
      this.hidden,
      this.logitsBuffer,
      this.tokens,
    ].reduce((total, buffer) => total + buffer.size, 0);

    this.layers = [];
    for (let index = 0; index < this.config.n_layer; index += 1) {
      const keys = scalar(n_ctx * n_embd);
      const values = scalar(n_ctx * n_embd);
      this.bytes.kv_cache += n_ctx * n_embd * 4 * 2;
      this.layers.push({
        keys,
        values,
        ln1: this.tensor(`h.${index}.ln_1.weight`).buffers[0],
        ln1bias: this.tensor(`h.${index}.ln_1.bias`).buffers[0],
        attnWeight: this.tensor(`h.${index}.attn.c_attn.weight`),
        attnBias: this.tensor(`h.${index}.attn.c_attn.bias`).buffers[0],
        projWeight: this.tensor(`h.${index}.attn.c_proj.weight`),
        projBias: this.tensor(`h.${index}.attn.c_proj.bias`).buffers[0],
        ln2: this.tensor(`h.${index}.ln_2.weight`).buffers[0],
        ln2bias: this.tensor(`h.${index}.ln_2.bias`).buffers[0],
        fcWeight: this.tensor(`h.${index}.mlp.c_fc.weight`),
        fcBias: this.tensor(`h.${index}.mlp.c_fc.bias`).buffers[0],
        mlpWeight: this.tensor(`h.${index}.mlp.c_proj.weight`),
        mlpBias: this.tensor(`h.${index}.mlp.c_proj.bias`).buffers[0],
      });
    }

    const uniformBuffer = () =>
      this.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    this.lnParams = uniformBuffer();
    writeUniform(this.device, this.lnParams, [
      [0, "u32", n_embd],
      [8, "f32", this.config.layer_norm_epsilon],
    ]);
    const mat = (nIn, nOut, gelu) => {
      const buffer = uniformBuffer();
      writeUniform(this.device, buffer, [
        [0, "u32", nIn],
        [4, "u32", nOut],
        [8, "u32", gelu ? 1 : 0],
      ]);
      return buffer;
    };
    this.attnParams = mat(n_embd, 3 * n_embd, false);
    this.projParams = mat(n_embd, n_embd, false);
    this.fcParams = mat(n_embd, 4 * n_embd, true);
    this.mlpParams = mat(4 * n_embd, n_embd, false);
    this.kvParams = uniformBuffer();
    writeUniform(this.device, this.kvParams, [[0, "u32", n_embd]]);
    this.residualParams = uniformBuffer();
    writeUniform(this.device, this.residualParams, [[0, "u32", n_embd]]);
    this.attentionParams = uniformBuffer();
    writeUniform(this.device, this.attentionParams, [
      [0, "u32", n_embd],
      [4, "u32", this.config.n_head],
      [8, "u32", n_embd / this.config.n_head],
      [12, "f32", 1 / Math.sqrt(n_embd / this.config.n_head)],
    ]);
    this.headParams = uniformBuffer();
    this.argmaxParams = uniformBuffer();
    writeUniform(this.device, this.argmaxParams, [[0, "u32", vocab_size]]);

    this.embedParams = uniformBuffer();
    const wte = this.tensor("wte.weight");
    this.splitRow =
      wte.buffers.length > 1 ? wte.rowsPerBuffer : this.config.vocab_size;
    writeUniform(this.device, this.embedParams, [
      [0, "u32", n_embd],
      [4, "u32", this.splitRow],
      [8, "u32", this.int8 ? n_embd / 64 : vocab_size],
    ]);
    writeUniform(this.device, this.headParams, [
      [0, "u32", n_embd],
      [4, "u32", vocab_size],
      [8, "u32", this.splitRow],
    ]);

    this.stepParams = [];
    this.finalParams = [];
    this.stepBindGroups = [];
    this.finalBindGroups = [];
    this.tokenStaging = [];
    for (let slot = 0; slot < this.depth; slot += 1) {
      this.stepParams.push(uniformBuffer());
      this.finalParams.push(uniformBuffer());
      this.stepBindGroups.push(
        this.device.createBindGroup({
          layout: this.layouts.step,
          entries: [
            entry(0, this.stepParams[slot]),
            entry(1, this.tokens),
          ],
        }),
      );
      this.finalBindGroups.push(
        this.device.createBindGroup({
          layout: this.layouts.step,
          entries: [
            entry(0, this.finalParams[slot]),
            entry(1, this.tokens),
          ],
        }),
      );
      const staging = this.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      this.tokenStaging.push(staging);
      this.bytes.staging += 4;
    }
    this.logitsStaging = this.device.createBuffer({
      size: align(vocab_size * 4, 16),
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.bytes.staging += this.logitsStaging.size;

    this.keys = scalar(vocab_size);
    this.history = scalar(vocab_size);
    this.histogram = scalar(65536);
    this.selectState = scalar(4);
    this.candidateCount = scalar(4);
    this.candidates = scalar(256);
    this.bytes.activations +=
      this.keys.size +
      this.history.size +
      this.histogram.size +
      this.selectState.size +
      this.candidateCount.size +
      this.candidates.size;
    this.sampleParams = [];
    for (let slot = 0; slot < this.depth; slot += 1) {
      this.sampleParams.push(
        this.device.createBuffer({
          size: 32,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }),
      );
    }
  }

  bindGroup(layout, entries) {
    return this.device.createBindGroup({ layout, entries });
  }

  createPasses() {
    const layouts = this.layouts;
    const pipelines = this.pipelines;
    const n_embd = this.config.n_embd;
    const n_head = this.config.n_head;
    const vocab = this.config.vocab_size;
    const width = n_embd / 4;
    const wte = this.tensor("wte.weight");
    const lnF = this.tensor("ln_f.weight").buffers[0];
    const lnFbias = this.tensor("ln_f.bias").buffers[0];
    this.tokenPasses = [];

    const lnPass = (input, weight, bias) => ({
      pipeline: pipelines.layerNorm,
      bindGroup: this.bindGroup(layouts.layerNorm, [
        entry(0, input),
        entry(1, weight),
        entry(2, bias),
        entry(3, this.norm),
        entry(4, this.lnParams),
      ]),
      width: null,
      heightRows: true,
    });

    const linearPass = (tensor, bias, input, output, params) => {
      if (tensor.n_out % 4 !== 0) {
        throw new Error(`tensor with ${tensor.n_out} outputs is not a multiple of four`);
      }
      const width = Math.ceil(tensor.n_out / 4 / this.linearWorkgroup);
      if (tensor.int8) {
        return {
          pipeline: pipelines.linear,
          bindGroup: this.bindGroup(layouts.linearInt8, [
            entry(0, tensor.buffers[0]),
            entry(1, tensor.scales[0]),
            entry(2, bias),
            entry(3, input),
            entry(4, output),
            entry(5, params),
          ]),
          width,
          decodePipeline: pipelines.linearDecode,
          decodeWidth: tensor.n_out,
          heightRows: true,
        };
      }
      return {
        pipeline: pipelines.linear,
        bindGroup: this.bindGroup(layouts.linearF32, [
          entry(0, tensor.buffers[0]),
          entry(1, bias),
          entry(2, input),
          entry(3, output),
          entry(4, params),
        ]),
        width,
        decodePipeline: pipelines.linearDecode,
        decodeWidth: tensor.n_out,
        heightRows: true,
      };
    };

    this.tokenPasses.push({
      pipeline: pipelines.embed,
      bindGroup: this.int8
        ? this.bindGroup(layouts.embedInt8, [
            entry(0, wte.buffers[0]),
            entry(1, wte.buffers[1] ?? this.dummy),
            entry(2, wte.scales[0]),
            entry(3, wte.scales[1] ?? this.dummy),
            entry(4, this.tensor("wpe.weight").buffers[0]),
            entry(5, this.x),
            entry(6, this.embedParams),
          ])
        : this.bindGroup(layouts.embedF32, [
            entry(0, wte.buffers[0]),
            entry(1, wte.buffers[1] ?? this.dummy),
            entry(2, this.tensor("wpe.weight").buffers[0]),
            entry(3, this.x),
            entry(4, this.embedParams),
          ]),
      width,
      heightRows: true,
    });

    for (const layer of this.layers) {
      this.tokenPasses.push(lnPass(this.x, layer.ln1, layer.ln1bias));
      this.tokenPasses.push(
        linearPass(
          layer.attnWeight,
          layer.attnBias,
          this.norm,
          this.qkv,
          this.attnParams,
        ),
      );
      this.tokenPasses.push({
        pipeline: pipelines.kv,
        bindGroup: this.bindGroup(layouts.kv, [
          entry(0, this.qkv),
          entry(1, layer.keys),
          entry(2, layer.values),
          entry(3, this.kvParams),
        ]),
        width,
        heightRows: true,
      });
      this.tokenPasses.push({
        pipeline: pipelines.attention,
        bindGroup: this.bindGroup(layouts.attention, [
          entry(0, this.qkv),
          entry(1, layer.keys),
          entry(2, layer.values),
          entry(3, this.attn),
          entry(4, this.attentionParams),
        ]),
        width: n_head,
        heightRows: true,
      });
      this.tokenPasses.push(
        linearPass(
          layer.projWeight,
          layer.projBias,
          this.attn,
          this.projection,
          this.projParams,
        ),
      );
      this.tokenPasses.push({
        pipeline: pipelines.residual,
        bindGroup: this.bindGroup(layouts.residual, [
          entry(0, this.x),
          entry(1, this.projection),
          entry(2, this.residualParams),
        ]),
        width,
        heightRows: true,
      });
      this.tokenPasses.push(lnPass(this.x, layer.ln2, layer.ln2bias));
      this.tokenPasses.push(
        linearPass(
          layer.fcWeight,
          layer.fcBias,
          this.norm,
          this.hidden,
          this.fcParams,
        ),
      );
      this.tokenPasses.push(
        linearPass(
          layer.mlpWeight,
          layer.mlpBias,
          this.hidden,
          this.projection,
          this.mlpParams,
        ),
      );
      this.tokenPasses.push({
        pipeline: pipelines.residual,
        bindGroup: this.bindGroup(layouts.residual, [
          entry(0, this.x),
          entry(1, this.projection),
          entry(2, this.residualParams),
        ]),
        width,
        heightRows: true,
      });
    }

    this.finalPass = {
      pipeline: pipelines.layerNorm,
      bindGroup: this.bindGroup(layouts.layerNorm, [
        entry(0, this.x),
        entry(1, lnF),
        entry(2, lnFbias),
        entry(3, this.norm),
        entry(4, this.lnParams),
      ]),
    };
    this.headPass = {
      pipeline: pipelines.head,
      bindGroup: this.int8
        ? this.bindGroup(layouts.headInt8, [
            entry(0, wte.buffers[0]),
            entry(1, wte.buffers[1] ?? this.dummy),
            entry(2, wte.scales[0]),
            entry(3, wte.scales[1] ?? this.dummy),
            entry(4, this.norm),
            entry(5, this.logitsBuffer),
            entry(6, this.headParams),
          ])
        : this.bindGroup(layouts.headF32, [
            entry(0, wte.buffers[0]),
            entry(1, wte.buffers[1] ?? this.dummy),
            entry(2, this.norm),
            entry(3, this.logitsBuffer),
            entry(4, this.headParams),
          ]),
      width: vocab,
    };
    this.argmaxPass = {
      pipeline: pipelines.argmax,
      bindGroup: this.bindGroup(layouts.argmax, [
        entry(0, this.logitsBuffer),
        entry(1, this.argmaxParams),
      ]),
      width: 1,
    };

    this.sampleResetGroup = this.bindGroup(layouts.sampleReset, [
      entry(0, this.histogram),
    ]);
    this.sampleGroups = [];
    for (let slot = 0; slot < this.depth; slot += 1) {
      const params = this.sampleParams[slot];
      this.sampleGroups.push({
        penalty: this.bindGroup(layouts.samplePenalty, [
          entry(0, this.logitsBuffer),
          entry(1, this.history),
          entry(2, this.histogram),
          entry(3, this.keys),
          entry(4, params),
        ]),
        bucket: this.bindGroup(layouts.sampleBucket, [
          entry(0, this.histogram),
          entry(1, this.selectState),
          entry(2, params),
        ]),
        gather: this.bindGroup(layouts.sampleGather, [
          entry(0, this.keys),
          entry(1, this.selectState),
          entry(2, this.candidates),
          entry(3, this.candidateCount),
          entry(4, params),
        ]),
        token: this.bindGroup(layouts.sampleToken, [
          entry(0, this.logitsBuffer),
          entry(1, this.candidates),
          entry(2, this.candidateCount),
          entry(3, this.history),
          entry(4, params),
        ]),
      });
    }
    this.sampleWorkgroups = Math.ceil(vocab / 256);
  }

  writeStep(slot, start, rows) {
    const main = new ArrayBuffer(16);
    const mainView = new DataView(main);
    mainView.setUint32(0, start, true);
    mainView.setUint32(4, rows, true);
    this.device.queue.writeBuffer(this.stepParams[slot], 0, main);

    const final = new ArrayBuffer(16);
    const finalView = new DataView(final);
    finalView.setUint32(0, start, true);
    finalView.setUint32(4, rows, true);
    finalView.setUint32(8, rows - 1, true);
    this.device.queue.writeBuffer(this.finalParams[slot], 0, final);
  }

  encodeForward(encoder, slot, rows, includeArgmax, includeSample = false) {
    const pass = encoder.beginComputePass();
    const stepGroup = this.stepBindGroups[slot];
    const run = (item, x, y, group) => {
      const decode = rows === 1 && item.decodePipeline;
      pass.setPipeline(decode ? item.decodePipeline : item.pipeline);
      pass.setBindGroup(0, group ?? stepGroup);
      pass.setBindGroup(1, item.bindGroup);
      pass.dispatchWorkgroups(decode ? item.decodeWidth : x, y, 1);
    };
    for (const item of this.tokenPasses) {
      run(item, item.width ?? rows, item.heightRows ? rows : 1);
    }
    run(this.finalPass, 1, 1, this.finalBindGroups[slot]);
    run(this.headPass, this.headPass.width, 1);
    if (includeSample) {
      const groups = this.sampleGroups[slot];
      const sample = (pipeline, group, workgroups) => {
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, stepGroup);
        pass.setBindGroup(1, group);
        pass.dispatchWorkgroups(workgroups, 1, 1);
      };
      sample(this.pipelines.sampleReset, this.sampleResetGroup, 256);
      sample(this.pipelines.samplePenalty, groups.penalty, this.sampleWorkgroups);
      sample(this.pipelines.sampleBucket, groups.bucket, 1);
      sample(this.pipelines.sampleGather, groups.gather, this.sampleWorkgroups);
      sample(this.pipelines.sampleToken, groups.token, 1);
    } else if (includeArgmax) {
      run(this.argmaxPass, 1, 1);
    }
    pass.end();
  }

  writeSample(slot, counter, sampling) {
    const bytes = new ArrayBuffer(32);
    const view = new DataView(bytes);
    view.setFloat32(0, sampling.temperature, true);
    view.setUint32(4, Math.min(Math.max(Math.round(sampling.topK), 1), 256), true);
    view.setFloat32(8, sampling.topP, true);
    view.setFloat32(12, sampling.repetitionPenalty, true);
    view.setUint32(16, sampling.seed >>> 0, true);
    view.setUint32(20, counter, true);
    view.setUint32(24, this.config.vocab_size, true);
    view.setUint32(28, 256, true);
    this.device.queue.writeBuffer(this.sampleParams[slot], 0, bytes);
  }

  writeTokens(ids) {
    this.device.queue.writeBuffer(this.tokens, 0, Uint32Array.from(ids));
  }

  async logits(ids) {
    if (ids.length === 0 || ids.length > this.config.n_ctx) {
      throw new Error(`invalid prompt length ${ids.length}`);
    }
    this.writeTokens(ids);
    this.writeStep(0, 0, ids.length);
    const encoder = this.device.createCommandEncoder();
    this.encodeForward(encoder, 0, ids.length, false);
    encoder.copyBufferToBuffer(
      this.logitsBuffer,
      0,
      this.logitsStaging,
      0,
      this.config.vocab_size * 4,
    );
    this.device.queue.submit([encoder.finish()]);
    await this.logitsStaging.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(
      this.logitsStaging.getMappedRange(0, this.config.vocab_size * 4),
    ).slice();
    this.logitsStaging.unmap();
    return values;
  }

  async generate(ids, maxNewTokens, onToken, sampling = null) {
    if (ids.length === 0) {
      throw new Error("generate requires at least one token id");
    }
    if (ids.length + maxNewTokens > this.config.n_ctx) {
      throw new Error("prompt and generated tokens exceed the context length");
    }
    if (maxNewTokens <= 0) {
      return;
    }
    this.writeTokens(ids);
    if (sampling) {
      const counts = new Uint32Array(this.config.vocab_size);
      for (const id of ids) {
        counts[id] += 1;
      }
      this.device.queue.writeBuffer(this.history, 0, counts);
    }
    const pending = [];
    let submitted = 0;
    const submit = () => {
      const index = submitted;
      const slot = index % this.depth;
      const start = index === 0 ? 0 : ids.length + index - 1;
      const rows = index === 0 ? ids.length : 1;
      this.writeStep(slot, start, rows);
      if (sampling) {
        this.writeSample(slot, index, sampling);
      }
      const encoder = this.device.createCommandEncoder();
      this.encodeForward(encoder, slot, rows, !sampling, sampling !== null);
      encoder.copyBufferToBuffer(
        this.tokens,
        (start + rows) * 4,
        this.tokenStaging[slot],
        0,
        4,
      );
      this.device.queue.submit([encoder.finish()]);
      const promise = this.tokenStaging[slot]
        .mapAsync(GPUMapMode.READ)
        .then(() => {
          const token = new Uint32Array(
            this.tokenStaging[slot].getMappedRange(0, 4),
          )[0];
          this.tokenStaging[slot].unmap();
          return token;
        });
      pending.push(promise);
      submitted += 1;
    };

    for (let index = 0; index < Math.min(this.depth, maxNewTokens); index += 1) {
      submit();
    }
    let stopped = false;
    while (pending.length > 0) {
      const token = await pending.shift();
      if (stopped) {
        continue;
      }
      if (onToken(token) === false) {
        stopped = true;
        continue;
      }
      if (submitted < maxNewTokens) {
        submit();
      }
    }
  }

  memory() {
    const total =
      this.bytes.weights +
      this.bytes.activations +
      this.bytes.kv_cache +
      this.bytes.staging;
    return {
      weight_bytes: this.bytes.weights,
      activation_bytes: this.bytes.activations,
      kv_cache_bytes: this.bytes.kv_cache,
      staging_bytes: this.bytes.staging,
      total_bytes: total,
    };
  }
}
