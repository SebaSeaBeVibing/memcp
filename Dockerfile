# Stage 1 — Planner: compute the dependency recipe
FROM rust:1.84-slim AS planner
WORKDIR /app
RUN cargo install cargo-chef
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# Stage 2 — Builder: cache deps then build the binary
FROM rust:1.84-slim AS builder
WORKDIR /app
RUN cargo install cargo-chef
COPY --from=planner /app/recipe.json recipe.json
# Cache dependencies separately from application code
RUN cargo chef cook --release --recipe-path recipe.json
COPY . .
RUN cargo build --release

# Stage 3 — Runtime: minimal image with only the binary
FROM debian:bookworm-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/memcp /usr/local/bin/memcp
ENTRYPOINT ["memcp"]
