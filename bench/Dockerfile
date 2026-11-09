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

FROM base AS app

COPY Cargo.toml Cargo.lock ./
COPY crates/bench/Cargo.toml crates/bench/Cargo.toml
COPY crates/engine/Cargo.toml crates/engine/Cargo.toml
RUN mkdir -p crates/engine/src crates/bench/src/bin \
    && touch crates/engine/src/lib.rs crates/bench/src/lib.rs \
        crates/bench/src/bin/bench.rs crates/bench/src/bin/worker.rs \
    && cargo fetch --locked

COPY . .
RUN cargo build --release --locked \
    && install -m 0755 target/release/bench target/release/worker /usr/local/bin/
