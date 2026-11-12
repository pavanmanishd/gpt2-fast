# bench

The protocol, prompts and reference outputs are described in `SPEC.md`. `run.sh` builds the
container image and runs `bench run` in it, `--cpuset-cpus=0-3 --memory=8g`, for the requested
number of sessions.

## Rival engines

Three outside engines run under the same protocol: llama.cpp (`bench/llamacpp/`), ONNX Runtime
(`bench/engines/onnxruntime/`) and CTranslate2 (`bench/engines/ctranslate2/`). Each worker takes
token ids and `max_new_tokens` on stdin, runs the prompt, then prints every greedy token the
moment it is chosen.

Pinned versions: `onnxruntime==1.31.1` (the C++ runtime and the Python package), `onnx==1.19.1`,
`protobuf==4.25.8`, `ml_dtypes==0.5.4`, `flatbuffers==25.12.19`, `PyYAML==6.0.2`,
`ctranslate2==4.6.2`, `transformers==4.46.3`, `torch==2.5.1`, `numpy==2.1.3`. CTranslate2 is
pinned to 4.6.2 because 4.6.3 and later pass `dtype=` to `transformers.from_pretrained`, which
4.46.3 does not accept. The GPT-2 model directory is `models/gpt2`; converted models are written
to `models/gpt2-onnx/` and `models/gpt2-ct2/`, both git-ignored.

Build the image and convert:

```sh
image=gpt2-fast-bench:$(git rev-parse --short HEAD)
docker build --platform linux/arm64 -t "$image" -f bench/Dockerfile .

docker run --rm --platform linux/arm64 -v "$PWD/models:/work/models" "$image" \
    bash -c 'cd /work && bench/engines/onnxruntime/convert.sh gpt2'
docker run --rm --platform linux/arm64 -v "$PWD/models:/work/models" "$image" \
    bash -c 'cd /work && bench/engines/ctranslate2/convert.sh gpt2'
```

`convert.sh gpt2` takes the model name, so `convert.sh gpt2-medium` works once `models/gpt2-medium`
exists.

### ONNX Runtime

`export.py` exports the Hugging Face checkpoint with the past key/value tensors as explicit graph
inputs and outputs, then quantizes it with `onnxruntime.quantization.quantize_dynamic`
(`QInt8`, per-channel). The f32 export matches PyTorch to `2e-4` on the first logits and `8e-5`
after a cache step; the int8 export differs by up to `9.8`.

`worker.cpp` is a C++ worker: 4 intra-op threads (from `sched_getaffinity`), 1 inter-op thread,
`ORT_ENABLE_ALL` graph optimizations, one token per `Session::Run`, and the present tensors of a
step fed straight back as the next step's past tensors (no copy). Greedy argmax scans the logits
from index 0 with a strict `>`, so the lowest index wins a tie, as in every other worker.

```sh
bench/run.sh onnx-f32 5 -- onnx-worker --model models/gpt2-onnx/model.onnx
bench/run.sh onnx-int8 5 --quantized -- onnx-worker --model models/gpt2-onnx/model-int8.onnx
```

### CTranslate2

`ct2-transformers-converter` produces `models/gpt2-ct2/float32` and `models/gpt2-ct2/int8`.
`worker.py` builds the prompt token strings from the converted `vocabulary.json`, calls
`Generator.generate_batch(..., include_prompt_in_result=False, sampling_topk=1, end_token=[])` and
writes each token from the per-token callback. With `include_prompt_in_result=False` the prompt is
forwarded in one batch, minus its last token, which starts the decoding loop; `max_length` is then
the number of generated tokens. The callback runs inline in the C++ decoding loop, before the next
decoder step, so the bench's per-token timestamps are the decoder's own step times. The Python work
inside the callback (formatting and writing the JSON line) measures 0.014 ms per token, 0.07% of a
20 ms f32 token and 0.13% of an 8.4 ms int8 token.

```sh
bench/run.sh ct2-f32 5 -- python3 bench/engines/ctranslate2/worker.py \
    --model-dir models/gpt2-ct2/float32 --compute-type float32
bench/run.sh ct2-int8 5 --quantized -- python3 bench/engines/ctranslate2/worker.py \
    --model-dir models/gpt2-ct2/int8 --compute-type int8 --threads 2
```

`intra_threads` defaults to the affinity mask (4). The int8 model runs with `--threads 2`: on this
CPU CTranslate2 4.6.2's Ruy int8 path drops from about 136 tok/s at 2 threads to 22 tok/s at 4
threads, measured on `p256` with the same callback protocol. The f32 model does not have that
problem (57 tok/s at 4 threads, 20 at 2).
