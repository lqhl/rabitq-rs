FROM ann-benchmarks

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install maturin for building Python bindings
RUN pip install maturin

# Copy source code
WORKDIR /home/app/rabitq-rs
COPY . .

# Build and install Python package with PyO3 bindings
RUN maturin build --release --features python && \
    pip install target/wheels/rabitq_rs-*.whl

# Install the ann-benchmarks wrapper
RUN pip install -e .
