FROM rust:1.98.1-bookworm AS base

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv linux-perf \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

COPY reference/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

RUN cargo install inferno --version 0.12.8 --locked

WORKDIR /work

FROM base AS llama

ARG LLAMA_CPP_TAG=b11912

RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "$LLAMA_CPP_TAG" https://github.com/ggml-org/llama.cpp /opt/llama.cpp

# armv8.6-a+dotprod+i8mm matches the Apple M4 host (linux/arm64): the release
# build must not depend on -march=native of whatever machine builds the image.
RUN cmake -S /opt/llama.cpp -B /opt/llama.cpp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DGGML_NATIVE=OFF \
        -DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_BUILD_APP=OFF \
    && cmake --build /opt/llama.cpp/build -j"$(nproc)" \
    && cmake --install /opt/llama.cpp/build \
    && ldconfig

COPY bench/llamacpp /opt/llama-worker
RUN cmake -S /opt/llama-worker -B /opt/llama-worker/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH=/usr/local \
    && cmake --build /opt/llama-worker/build --target llama-worker -j"$(nproc)" \
    && install -m 0755 /opt/llama-worker/build/llama-worker /usr/local/bin/llama-worker

FROM base AS onnx

ARG ONNXRUNTIME_VERSION=1.31.1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz" \
        -o /tmp/onnxruntime.tgz \
    && tar -xzf /tmp/onnxruntime.tgz -C /opt \
    && mv "/opt/onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}" /opt/onnxruntime \
    && rm /tmp/onnxruntime.tgz \
    && cp -a /opt/onnxruntime/lib/libonnxruntime.so* /usr/local/lib/ \
    && cp -a /opt/onnxruntime/include/* /usr/local/include/ \
    && ldconfig

COPY bench/engines/onnxruntime /opt/onnx-worker

RUN g++ -O3 -march=armv8.6-a+dotprod+i8mm -std=c++17 \
        -I/usr/local/include /opt/onnx-worker/worker.cpp \
        -L/usr/local/lib -lonnxruntime \
        -Wl,-rpath,/usr/local/lib \
        -o /usr/local/bin/onnx-worker

FROM base AS app

# onnx 1.19.1, onnxruntime 1.31.1, protobuf 4.25.8, ml_dtypes 0.5.4,
# flatbuffers 25.12.19, PyYAML 6.0.2, ctranslate2 4.6.2
RUN pip install --no-cache-dir \
        numpy==2.1.3 \
        onnx==1.19.1 \
        onnxruntime==1.31.1 \
        protobuf==4.25.8 \
        ml_dtypes==0.5.4 \
        flatbuffers==25.12.19 \
        PyYAML==6.0.2 \
        ctranslate2==4.6.2

COPY --from=llama /usr/local/bin/llama-worker /usr/local/bin/llama-worker
COPY --from=llama /usr/local/bin/llama-quantize /usr/local/bin/llama-quantize
COPY --from=llama /usr/local/lib/libllama*.so* /usr/local/lib/
COPY --from=llama /usr/local/lib/libggml*.so* /usr/local/lib/
COPY --from=llama /opt/llama.cpp/convert_hf_to_gguf.py /opt/llama.cpp/convert_hf_to_gguf.py
COPY --from=llama /opt/llama.cpp/conversion /opt/llama.cpp/conversion
COPY --from=llama /opt/llama.cpp/gguf-py /opt/llama.cpp/gguf-py
RUN ldconfig \
    && pip install --no-cache-dir --no-deps /opt/llama.cpp/gguf-py

COPY --from=onnx /usr/local/bin/onnx-worker /usr/local/bin/onnx-worker
COPY --from=onnx /usr/local/lib/libonnxruntime.so* /usr/local/lib/
RUN ldconfig

COPY Cargo.toml Cargo.lock ./
COPY crates/bench/Cargo.toml crates/bench/Cargo.toml
COPY crates/engine/Cargo.toml crates/engine/Cargo.toml
COPY crates/web/Cargo.toml crates/web/Cargo.toml
RUN mkdir -p crates/engine/src crates/bench/src/bin crates/web/src \
    && touch crates/engine/src/lib.rs crates/bench/src/lib.rs \
        crates/bench/src/bin/bench.rs crates/bench/src/bin/worker.rs \
        crates/web/src/lib.rs \
    && cargo fetch --locked

COPY . .
RUN cargo build --release --locked -p gpt2_bench \
    && install -m 0755 target/release/bench target/release/worker /usr/local/bin/
