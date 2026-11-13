const STEP = `
struct Step {
  start: u32,
  n: u32,
  first_row: u32,
  pad: u32,
};

@group(0) @binding(0) var<uniform> step: Step;
@group(0) @binding(1) var<storage, read_write> tokens: array<u32>;
`;

const SAMPLE = `
struct SampleParams {
  temperature: f32,
  top_k: u32,
  top_p: f32,
  penalty: f32,
  seed: u32,
  counter: u32,
  vocab: u32,
  capacity: u32,
};
`;

const GELU = `
fn gelu(value: f32) -> f32 {
  let inner = 0.7978845608028654 * (value + 0.044715 * value * value * value);
  // Metal's tanh overflows to NaN for large arguments, and tanh saturates long before this.
  return 0.5 * value * (1.0 + tanh(clamp(inner, -15.0, 15.0)));
}
`;

export const embedF32 = `
struct EmbedParams {
  n_embd: u32,
  split_row: u32,
  vocab: u32,
  pad: u32,
};

${STEP}

@group(1) @binding(0) var<storage, read> wte0: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> wte1: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read> wpe: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> output: array<vec4<f32>>;
@group(1) @binding(4) var<uniform> params: EmbedParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let width = params.n_embd / 4u;
  let chunk = gid.x;
  let row = gid.y;
  if (chunk >= width || row >= step.n) {
    return;
  }
  let position = step.start + row;
  let id = tokens[position];
  var token_row: vec4<f32>;
  if (id < params.split_row) {
    token_row = wte0[id * width + chunk];
  } else {
    token_row = wte1[(id - params.split_row) * width + chunk];
  }
  output[row * width + chunk] = token_row + wpe[position * width + chunk];
}
`;

export const embedInt8 = `
struct EmbedParams {
  n_embd: u32,
  split_row: u32,
  groups: u32,
  pad: u32,
};

${STEP}

@group(1) @binding(0) var<storage, read> wte0: array<u32>;
@group(1) @binding(1) var<storage, read> wte1: array<u32>;
@group(1) @binding(2) var<storage, read> scales0: array<f32>;
@group(1) @binding(3) var<storage, read> scales1: array<f32>;
@group(1) @binding(4) var<storage, read> wpe: array<vec4<f32>>;
@group(1) @binding(5) var<storage, read_write> output: array<vec4<f32>>;
@group(1) @binding(6) var<uniform> params: EmbedParams;

fn unpack4(word: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(bitcast<i32>(word << 24u) >> 24),
    f32(bitcast<i32>(word << 16u) >> 24),
    f32(bitcast<i32>(word << 8u) >> 24),
    f32(bitcast<i32>(word) >> 24),
  );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let width = params.n_embd / 4u;
  let chunk = gid.x;
  let row = gid.y;
  if (chunk >= width || row >= step.n) {
    return;
  }
  let position = step.start + row;
  let id = tokens[position];
  let group = chunk / 16u;
  var word: u32;
  var scale: f32;
  if (id < params.split_row) {
    word = wte0[id * width + chunk];
    scale = scales0[id * params.groups + group];
  } else {
    word = wte1[(id - params.split_row) * width + chunk];
    scale = scales1[(id - params.split_row) * params.groups + group];
  }
  output[row * width + chunk] = unpack4(word) * scale + wpe[position * width + chunk];
}
`;

export const linearF32 = `
struct Mat {
  n_in: u32,
  n_out: u32,
  flags: u32,
  pad: u32,
};

${STEP}
${GELU}

@group(1) @binding(0) var<storage, read> weights: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> bias: array<f32>;
@group(1) @binding(2) var<storage, read> input: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> output: array<vec4<f32>>;
@group(1) @binding(4) var<uniform> params: Mat;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.y;
  if (row >= step.n) {
    return;
  }
  let n_in4 = params.n_in / 4u;
  let n_out4 = params.n_out / 4u;
  let group = gid.x;
  if (group >= n_out4) {
    return;
  }
  let weight_row = group * 4u * n_in4;
  let input_row = row * n_in4;
  var acc0 = 0.0;
  var acc1 = 0.0;
  var acc2 = 0.0;
  var acc3 = 0.0;
  for (var p = 0u; p < n_in4; p = p + 1u) {
    let x = input[input_row + p];
    acc0 = acc0 + dot(x, weights[weight_row + p]);
    acc1 = acc1 + dot(x, weights[weight_row + n_in4 + p]);
    acc2 = acc2 + dot(x, weights[weight_row + 2u * n_in4 + p]);
    acc3 = acc3 + dot(x, weights[weight_row + 3u * n_in4 + p]);
  }
  let base = group * 4u;
  var value = vec4<f32>(
    acc0 + bias[base],
    acc1 + bias[base + 1u],
    acc2 + bias[base + 2u],
    acc3 + bias[base + 3u],
  );
  if ((params.flags & 1u) != 0u) {
    value = vec4<f32>(gelu(value.x), gelu(value.y), gelu(value.z), gelu(value.w));
  }
  output[row * n_out4 + group] = value;
}
`;

export const linearInt8 = `
struct Mat {
  n_in: u32,
  n_out: u32,
  flags: u32,
  pad: u32,
};

${STEP}
${GELU}

@group(1) @binding(0) var<storage, read> weights: array<u32>;
@group(1) @binding(1) var<storage, read> scales: array<f32>;
@group(1) @binding(2) var<storage, read> bias: array<f32>;
@group(1) @binding(3) var<storage, read> input: array<vec4<f32>>;
@group(1) @binding(4) var<storage, read_write> output: array<vec4<f32>>;
@group(1) @binding(5) var<uniform> params: Mat;

fn unpack4(word: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(bitcast<i32>(word << 24u) >> 24),
    f32(bitcast<i32>(word << 16u) >> 24),
    f32(bitcast<i32>(word << 8u) >> 24),
    f32(bitcast<i32>(word) >> 24),
  );
}

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.y;
  if (row >= step.n) {
    return;
  }
  let n_in4 = params.n_in / 4u;
  let n_out4 = params.n_out / 4u;
  let group = gid.x;
  if (group >= n_out4) {
    return;
  }
  let groups = params.n_in / 64u;
  let weight_row = group * 4u * n_in4;
  let input_row = row * n_in4;
  var total0 = 0.0;
  var total1 = 0.0;
  var total2 = 0.0;
  var total3 = 0.0;
  for (var g = 0u; g < groups; g = g + 1u) {
    var acc0 = 0.0;
    var acc1 = 0.0;
    var acc2 = 0.0;
    var acc3 = 0.0;
    let first = g * 16u;
    for (var p = first; p < first + 16u; p = p + 1u) {
      let x = input[input_row + p];
      acc0 = acc0 + dot(x, unpack4(weights[weight_row + p]));
      acc1 = acc1 + dot(x, unpack4(weights[weight_row + n_in4 + p]));
      acc2 = acc2 + dot(x, unpack4(weights[weight_row + 2u * n_in4 + p]));
      acc3 = acc3 + dot(x, unpack4(weights[weight_row + 3u * n_in4 + p]));
    }
    let base = group * 4u;
    total0 = total0 + acc0 * scales[(base + 0u) * groups + g];
    total1 = total1 + acc1 * scales[(base + 1u) * groups + g];
    total2 = total2 + acc2 * scales[(base + 2u) * groups + g];
    total3 = total3 + acc3 * scales[(base + 3u) * groups + g];
  }
  let base = group * 4u;
  var value = vec4<f32>(
    total0 + bias[base],
    total1 + bias[base + 1u],
    total2 + bias[base + 2u],
    total3 + bias[base + 3u],
  );
  if ((params.flags & 1u) != 0u) {
    value = vec4<f32>(gelu(value.x), gelu(value.y), gelu(value.z), gelu(value.w));
  }
  output[row * n_out4 + group] = value;
}
`;

export const linearDecodeF32 = `
struct Mat {
  n_in: u32,
  n_out: u32,
  flags: u32,
  pad: u32,
};

${STEP}
${GELU}

@group(1) @binding(0) var<storage, read> weights: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> bias: array<f32>;
@group(1) @binding(2) var<storage, read> input: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> output: array<f32>;
@group(1) @binding(4) var<uniform> params: Mat;

var<workgroup> reduce: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) workgroup: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let out_index = workgroup.x;
  if (out_index >= params.n_out) {
    return;
  }
  let n_in4 = params.n_in / 4u;
  let weight_row = out_index * n_in4;
  var sum = 0.0;
  for (var p = local.x; p < n_in4; p = p + 64u) {
    sum = sum + dot(input[p], weights[weight_row + p]);
  }
  reduce[local.x] = sum;
  workgroupBarrier();
  var stride = 32u;
  while (stride > 0u) {
    if (local.x < stride) {
      reduce[local.x] = reduce[local.x] + reduce[local.x + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (local.x == 0u) {
    var value = reduce[0] + bias[out_index];
    if ((params.flags & 1u) != 0u) {
      value = gelu(value);
    }
    output[out_index] = value;
  }
}
`;

export const linearDecodeInt8 = `
struct Mat {
  n_in: u32,
  n_out: u32,
  flags: u32,
  pad: u32,
};

${STEP}
${GELU}

@group(1) @binding(0) var<storage, read> weights: array<u32>;
@group(1) @binding(1) var<storage, read> scales: array<f32>;
@group(1) @binding(2) var<storage, read> bias: array<f32>;
@group(1) @binding(3) var<storage, read> input: array<vec4<f32>>;
@group(1) @binding(4) var<storage, read_write> output: array<f32>;
@group(1) @binding(5) var<uniform> params: Mat;

var<workgroup> reduce: array<f32, 64>;

fn unpack4(word: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(bitcast<i32>(word << 24u) >> 24),
    f32(bitcast<i32>(word << 16u) >> 24),
    f32(bitcast<i32>(word << 8u) >> 24),
    f32(bitcast<i32>(word) >> 24),
  );
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) workgroup: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let out_index = workgroup.x;
  if (out_index >= params.n_out) {
    return;
  }
  let n_in4 = params.n_in / 4u;
  let groups = params.n_in / 64u;
  let lane = local.x;
  let slot = lane % 16u;
  let within = lane / 16u;
  var total = 0.0;
  for (var base = 0u; base < groups; base = base + 4u) {
    let group = base + within;
    if (group < groups) {
      let position = group * 16u + slot;
      total = total + dot(input[position], unpack4(weights[out_index * n_in4 + position])) * scales[out_index * groups + group];
    }
  }
  reduce[lane] = total;
  workgroupBarrier();
  var stride = 32u;
  while (stride > 0u) {
    if (lane < stride) {
      reduce[lane] = reduce[lane] + reduce[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (lane == 0u) {
    var value = reduce[0] + bias[out_index];
    if ((params.flags & 1u) != 0u) {
      value = gelu(value);
    }
    output[out_index] = value;
  }
}
`;

export const headDecodeF32 = `
struct HeadParams {
  n_in: u32,
  n_out: u32,
  split_row: u32,
  pad: u32,
};

${STEP}

@group(1) @binding(0) var<storage, read> wte0: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> wte1: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read> input: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> output: array<f32>;
@group(1) @binding(4) var<uniform> params: HeadParams;

var<workgroup> reduce: array<f32, 64>;

fn token_row(index: u32, position: u32) -> vec4<f32> {
  let n_in4 = params.n_in / 4u;
  let row = min(index, params.n_out - 1u);
  if (row < params.split_row) {
    return wte0[row * n_in4 + position];
  }
  return wte1[(row - params.split_row) * n_in4 + position];
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) workgroup: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let out_index = workgroup.x;
  if (out_index >= params.n_out) {
    return;
  }
  let n_in4 = params.n_in / 4u;
  var sum = 0.0;
  for (var p = local.x; p < n_in4; p = p + 64u) {
    sum = sum + dot(input[p], token_row(out_index, p));
  }
  reduce[local.x] = sum;
  workgroupBarrier();
  var stride = 32u;
  while (stride > 0u) {
    if (local.x < stride) {
      reduce[local.x] = reduce[local.x] + reduce[local.x + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (local.x == 0u) {
    output[out_index] = reduce[0];
  }
}
`;

export const headDecodeInt8 = `
struct HeadParams {
  n_in: u32,
  n_out: u32,
  split_row: u32,
  pad: u32,
};

${STEP}

@group(1) @binding(0) var<storage, read> wte0: array<u32>;
@group(1) @binding(1) var<storage, read> wte1: array<u32>;
@group(1) @binding(2) var<storage, read> scales0: array<f32>;
@group(1) @binding(3) var<storage, read> scales1: array<f32>;
@group(1) @binding(4) var<storage, read> input: array<vec4<f32>>;
@group(1) @binding(5) var<storage, read_write> output: array<f32>;
@group(1) @binding(6) var<uniform> params: HeadParams;

var<workgroup> reduce: array<f32, 64>;

fn unpack4(word: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(bitcast<i32>(word << 24u) >> 24),
    f32(bitcast<i32>(word << 16u) >> 24),
    f32(bitcast<i32>(word << 8u) >> 24),
    f32(bitcast<i32>(word) >> 24),
  );
}

fn token_word(index: u32, position: u32) -> u32 {
  let n_in4 = params.n_in / 4u;
  let row = min(index, params.n_out - 1u);
  if (row < params.split_row) {
    return wte0[row * n_in4 + position];
  }
  return wte1[(row - params.split_row) * n_in4 + position];
}

fn token_scale(index: u32, group: u32) -> f32 {
  let groups = params.n_in / 64u;
  let row = min(index, params.n_out - 1u);
  if (row < params.split_row) {
    return scales0[row * groups + group];
  }
  return scales1[(row - params.split_row) * groups + group];
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) workgroup: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let out_index = workgroup.x;
  if (out_index >= params.n_out) {
    return;
  }
  let groups = params.n_in / 64u;
  let lane = local.x;
  let slot = lane % 16u;
  let within = lane / 16u;
  var total = 0.0;
  for (var base = 0u; base < groups; base = base + 4u) {
    let group = base + within;
    if (group < groups) {
      let position = group * 16u + slot;
      total = total + dot(input[position], unpack4(token_word(out_index, position))) * token_scale(out_index, group);
    }
  }
  reduce[lane] = total;
  workgroupBarrier();
  var stride = 32u;
  while (stride > 0u) {
    if (lane < stride) {
      reduce[lane] = reduce[lane] + reduce[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (lane == 0u) {
    output[out_index] = reduce[0];
  }
}
`;

export const layerNorm = `
struct LnParams {
  n_embd: u32,
  pad: u32,
  eps: f32,
  pad2: f32,
};

${STEP}

@group(1) @binding(0) var<storage, read> input: array<f32>;
@group(1) @binding(1) var<storage, read> weight: array<f32>;
@group(1) @binding(2) var<storage, read> bias: array<f32>;
@group(1) @binding(3) var<storage, read_write> output: array<f32>;
@group(1) @binding(4) var<uniform> params: LnParams;

var<workgroup> reduce: array<f32, 256>;

fn block_sum(lane: u32) -> f32 {
  var stride = 128u;
  while (stride > 0u) {
    if (lane < stride) {
      reduce[lane] = reduce[lane] + reduce[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  return reduce[0];
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroup: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let lane = local.x;
  let input_row = step.first_row + workgroup.x;
  let output_row = workgroup.x;
  let n = params.n_embd;
  var sum = 0.0;
  for (var i = lane; i < n; i = i + 256u) {
    sum = sum + input[input_row * n + i];
  }
  reduce[lane] = sum;
  workgroupBarrier();
  let mean = block_sum(lane) / f32(n);
  var variance = 0.0;
  for (var i = lane; i < n; i = i + 256u) {
    let centered = input[input_row * n + i] - mean;
    variance = variance + centered * centered;
  }
  reduce[lane] = variance;
  workgroupBarrier();
  let inverse = inverseSqrt(block_sum(lane) / f32(n) + params.eps);
  for (var i = lane; i < n; i = i + 256u) {
    output[output_row * n + i] = (input[input_row * n + i] - mean) * inverse * weight[i] + bias[i];
  }
}
`;

export const kvStore = `
struct KvParams {
  n_embd: u32,
  pad: u32,
  pad2: u32,
  pad3: u32,
};

${STEP}

@group(1) @binding(0) var<storage, read> qkv: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> keys: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> values: array<vec4<f32>>;
@group(1) @binding(3) var<uniform> params: KvParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let width = params.n_embd / 4u;
  let chunk = gid.x;
  let row = gid.y;
  if (chunk >= width || row >= step.n) {
    return;
  }
  let source = row * 3u * width;
  let destination = (step.start + row) * width;
  keys[destination + chunk] = qkv[source + width + chunk];
  values[destination + chunk] = qkv[source + 2u * width + chunk];
}
`;

export const attention = `
struct AttnParams {
  n_embd: u32,
  n_head: u32,
  head_dim: u32,
  scale: f32,
};

${STEP}

@group(1) @binding(0) var<storage, read> qkv: array<f32>;
@group(1) @binding(1) var<storage, read> keys: array<f32>;
@group(1) @binding(2) var<storage, read> values: array<f32>;
@group(1) @binding(3) var<storage, read_write> output: array<f32>;
@group(1) @binding(4) var<uniform> params: AttnParams;

var<workgroup> query: array<f32, 64>;
var<workgroup> scores: array<f32, 1024>;
var<workgroup> reduce: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) workgroup: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let head = workgroup.x;
  let row = workgroup.y;
  let lane = local.x;
  let dim = params.head_dim;
  let base = head * dim;
  let query_offset = row * 3u * params.n_embd + base;
  if (lane < dim) {
    query[lane] = qkv[query_offset + lane];
  }
  workgroupBarrier();

  let end = step.start + row + 1u;
  var local_max = -3.402823e38;
  var past = lane;
  while (past < end) {
    var sum = 0.0;
    let key_offset = past * params.n_embd + base;
    for (var d = 0u; d < dim; d = d + 1u) {
      sum = sum + query[d] * keys[key_offset + d];
    }
    let score = sum * params.scale;
    scores[past] = score;
    local_max = max(local_max, score);
    past = past + 64u;
  }
  reduce[lane] = local_max;
  workgroupBarrier();
  var stride = 32u;
  while (stride > 0u) {
    if (lane < stride) {
      reduce[lane] = max(reduce[lane], reduce[lane + stride]);
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let highest = reduce[0];

  var local_sum = 0.0;
  past = lane;
  while (past < end) {
    let weight = exp(scores[past] - highest);
    scores[past] = weight;
    local_sum = local_sum + weight;
    past = past + 64u;
  }
  reduce[lane] = local_sum;
  workgroupBarrier();
  stride = 32u;
  while (stride > 0u) {
    if (lane < stride) {
      reduce[lane] = reduce[lane] + reduce[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let total = reduce[0];

  if (lane < dim) {
    var acc = 0.0;
    for (var p = 0u; p < end; p = p + 1u) {
      acc = acc + (scores[p] / total) * values[p * params.n_embd + base + lane];
    }
    output[row * params.n_embd + base + lane] = acc;
  }
}
`;

export const residual = `
struct ResidualParams {
  n_embd: u32,
  pad: u32,
  pad2: u32,
  pad3: u32,
};

${STEP}

@group(1) @binding(0) var<storage, read_write> x: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> addend: array<vec4<f32>>;
@group(1) @binding(2) var<uniform> params: ResidualParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let width = params.n_embd / 4u;
  let chunk = gid.x;
  let row = gid.y;
  if (chunk >= width || row >= step.n) {
    return;
  }
  let index = row * width + chunk;
  x[index] = x[index] + addend[index];
}
`;



export const argmax = `
struct ArgParams {
  vocab: u32,
  pad: u32,
  pad2: u32,
  pad3: u32,
};

${STEP}

@group(1) @binding(0) var<storage, read> logits: array<f32>;
@group(1) @binding(1) var<uniform> params: ArgParams;

var<workgroup> best_value: array<f32, 256>;
var<workgroup> best_index: array<u32, 256>;

fn better(value: f32, index: u32, other_value: f32, other_index: u32) -> bool {
  return value > other_value || (value == other_value && index < other_index);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
  let lane = local.x;
  var value = -3.402823e38;
  var index = 0xffffffffu;
  for (var i = lane; i < params.vocab; i = i + 256u) {
    let candidate = logits[i];
    if (better(candidate, i, value, index)) {
      value = candidate;
      index = i;
    }
  }
  best_value[lane] = value;
  best_index[lane] = index;
  workgroupBarrier();
  var stride = 128u;
  while (stride > 0u) {
    if (lane < stride) {
      if (better(best_value[lane + stride], best_index[lane + stride], best_value[lane], best_index[lane])) {
        best_value[lane] = best_value[lane + stride];
        best_index[lane] = best_index[lane + stride];
      }
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (lane == 0u) {
    tokens[step.start + step.n] = best_index[0];
  }
}
`;

export const sampleReset = `
@group(1) @binding(0) var<storage, read_write> hist: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index < 65536u) {
    atomicStore(&hist[index], 0u);
  }
}
`;

export const samplePenalty = `
${STEP}
${SAMPLE}

@group(1) @binding(0) var<storage, read_write> logits: array<f32>;
@group(1) @binding(1) var<storage, read> counts: array<u32>;
@group(1) @binding(2) var<storage, read_write> hist: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read_write> keys: array<u32>;
@group(1) @binding(4) var<uniform> params: SampleParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.vocab) {
    return;
  }
  var value = logits[index];
  if (counts[index] > 0u) {
    if (value > 0.0) {
      value = value / params.penalty;
    } else {
      value = value * params.penalty;
    }
  }
  logits[index] = value;
  let word = bitcast<u32>(value);
  let key = select(word | 0x80000000u, ~word, (word & 0x80000000u) != 0u);
  keys[index] = key;
  atomicAdd(&hist[key >> 16u], 1u);
}
`;

export const sampleBucket = `
${SAMPLE}

@group(1) @binding(0) var<storage, read> hist: array<u32>;
@group(1) @binding(1) var<storage, read_write> state: array<u32>;
@group(1) @binding(2) var<uniform> params: SampleParams;

var<workgroup> regions: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
  let lane = local.x;
  var total = 0u;
  for (var index = 0u; index < 256u; index = index + 1u) {
    total = total + hist[lane * 256u + index];
  }
  regions[lane] = total;
  workgroupBarrier();
  if (lane == 0u) {
    let need = min(max(params.top_k, 1u), params.vocab);
    var cumulative = 0u;
    var boundary = 0u;
    var above = 0u;
    var found = false;
    for (var index = 0u; index < 256u; index = index + 1u) {
      let region = 255u - index;
      let count = regions[region];
      if (cumulative + count >= need) {
        var inner = 0u;
        for (var slot = 0u; slot < 256u; slot = slot + 1u) {
          let bin = 255u - slot;
          let in_bin = hist[region * 256u + bin];
          if (inner + in_bin >= need - cumulative) {
            boundary = region * 256u + bin;
            above = cumulative + inner;
            found = true;
            break;
          }
          inner = inner + in_bin;
        }
        if (found) {
          break;
        }
      }
      cumulative = cumulative + count;
    }
    state[0] = boundary;
    state[1] = above;
    state[2] = need;
  }
}
`;

export const sampleGather = `
${SAMPLE}

@group(1) @binding(0) var<storage, read> keys: array<u32>;
@group(1) @binding(1) var<storage, read> state: array<u32>;
@group(1) @binding(2) var<storage, read_write> candidates: array<u32>;
@group(1) @binding(3) var<storage, read_write> count: array<atomic<u32>>;
@group(1) @binding(4) var<uniform> params: SampleParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.vocab) {
    return;
  }
  if ((keys[index] >> 16u) >= state[0]) {
    let slot = atomicAdd(&count[0], 1u);
    if (slot < params.capacity) {
      candidates[slot] = index;
    }
  }
}
`;

export const sampleToken = `
${STEP}
${SAMPLE}

@group(1) @binding(0) var<storage, read> logits: array<f32>;
@group(1) @binding(1) var<storage, read> candidates: array<u32>;
@group(1) @binding(2) var<storage, read_write> count: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read_write> counts: array<atomic<u32>>;
@group(1) @binding(4) var<uniform> params: SampleParams;

var<workgroup> values: array<f32, 256>;
var<workgroup> indices: array<u32, 256>;
var<workgroup> sorted_values: array<f32, 256>;
var<workgroup> sorted_indices: array<u32, 256>;

fn better(value: f32, index: u32, other_value: f32, other_index: u32) -> bool {
  return value > other_value || (value == other_value && index < other_index);
}

fn draw(seed: u32, counter: u32) -> f32 {
  var x = seed ^ (0x9e3779b9u * (counter + 1u));
  x = x ^ (x >> 16u);
  x = x * 0x7feb352du;
  x = x ^ (x >> 15u);
  x = x * 0x846ca68bu;
  x = x ^ (x >> 16u);
  return f32(x >> 8u) * (1.0 / 16777216.0);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
  let lane = local.x;
  let available = min(atomicLoad(&count[0]), params.capacity);
  let limit = min(max(params.top_k, 1u), available);
  if (lane < available) {
    let token = candidates[lane];
    indices[lane] = token;
    values[lane] = logits[token];
  }
  workgroupBarrier();
  var chosen = 0u;
  if (lane < available) {
    var rank = 0u;
    for (var index = 0u; index < available; index = index + 1u) {
      if (better(values[index], indices[index], values[lane], indices[lane])) {
        rank = rank + 1u;
      }
    }
    if (rank < limit) {
      sorted_values[rank] = values[lane];
      sorted_indices[rank] = indices[lane];
    }
  }
  workgroupBarrier();
  if (lane == 0u && limit > 0u) {
    if (params.temperature <= 0.0) {
      chosen = sorted_indices[0];
    } else {
      let scale = 1.0 / params.temperature;
      let highest = sorted_values[0];
      var total = 0.0;
      for (var rank = 0u; rank < limit; rank = rank + 1u) {
        let probability = exp((sorted_values[rank] - highest) * scale);
        sorted_values[rank] = probability;
        total = total + probability;
      }
      var keep = limit;
      var cutoff = 1.0;
      if (params.top_p < 1.0) {
        var cumulative = 0.0;
        for (var rank = 0u; rank < limit; rank = rank + 1u) {
          cumulative = cumulative + sorted_values[rank] / total;
          keep = rank + 1u;
          if (cumulative >= params.top_p) {
            break;
          }
        }
        cutoff = cumulative;
      }
      let point = draw(params.seed, params.counter) * cutoff;
      var running = 0.0;
      chosen = sorted_indices[keep - 1u];
      for (var rank = 0u; rank < keep; rank = rank + 1u) {
        running = running + sorted_values[rank] / total;
        if (point < running) {
          chosen = sorted_indices[rank];
          break;
        }
      }
    }
    tokens[step.start + step.n] = chosen;
    atomicAdd(&counts[chosen], 1u);
    atomicStore(&count[0], 0u);
  }
}
`;
