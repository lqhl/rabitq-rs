# Ann-Benchmarks Algorithm Templates

This directory contains templates for integrating RaBitQ-RS algorithms with [ann-benchmarks](https://github.com/erikbern/ann-benchmarks).

## Structure

```
ann_benchmarks_templates/
├── rabitq-mstg/          # MSTG algorithm (Multi-Scale Tree Graph)
│   ├── Dockerfile        # Docker build instructions
│   ├── config.yml        # MSTG benchmark configurations
│   └── module.py         # Python wrapper for MSTG
└── rabitq-ivf/           # IVF algorithm (Inverted File Index)
    ├── Dockerfile        # Docker build instructions
    ├── config.yml        # IVF benchmark configurations
    └── module.py         # Python wrapper for IVF
```

## Deployment to ann-benchmarks

Each algorithm directory is **independent** and can be copied to the ann-benchmarks repository:

### Option 1: Manual Copy
```bash
# Copy MSTG
cp -r ann_benchmarks_templates/rabitq-mstg/ \
    /path/to/ann-benchmarks/ann_benchmarks/algorithms/

# Copy IVF
cp -r ann_benchmarks_templates/rabitq-ivf/ \
    /path/to/ann-benchmarks/ann_benchmarks/algorithms/
```

### Option 2: Symlink (for development)
```bash
# Link MSTG
ln -s $(pwd)/ann_benchmarks_templates/rabitq-mstg \
    /path/to/ann-benchmarks/ann_benchmarks/algorithms/

# Link IVF
ln -s $(pwd)/ann_benchmarks_templates/rabitq-ivf \
    /path/to/ann-benchmarks/ann_benchmarks/algorithms/
```

## Building Docker Images

From the ann-benchmarks repository:

```bash
# Build MSTG
python install.py --algorithm rabitq-mstg

# Build IVF
python install.py --algorithm rabitq-ivf
```

## Running Benchmarks

```bash
# Run MSTG on fashion-mnist
python run.py --algorithm rabitq-mstg --dataset fashion-mnist-784-euclidean

# Run IVF on GIST
python run.py --algorithm rabitq-ivf --dataset gist-960-euclidean
```

## Configuration Files

### rabitq-mstg/config.yml
Contains 5 pre-configured run groups:
- **base**: Balanced recall/latency (256 posting size)
- **high-recall**: 95%+ recall target (128 posting size)
- **low-latency**: Speed priority (512 posting size)
- **memory-efficient**: Minimal footprint (1024 posting size, 3-bit quantization)
- **billion-scale**: Optimized for 100M+ vectors (2048 posting size)

### rabitq-ivf/config.yml
Contains 8 nlist configurations (32, 64, 128, 256, 512, 1024, 2048, 4096) with varying nprobe settings.

## Testing Without Docker

You can test the configurations locally without Docker:

```bash
# Install rabitq_rs Python package first
maturin develop --release --features python

# Run the test script
python ../test_mstg_config.py /path/to/ann-benchmarks/data base
```

## Correspondence

| rabitq-rs template | ann-benchmarks location |
|-------------------|------------------------|
| `ann_benchmarks_templates/rabitq-mstg/` | `ann_benchmarks/algorithms/rabitq-mstg/` |
| `ann_benchmarks_templates/rabitq-ivf/` | `ann_benchmarks/algorithms/rabitq-ivf/` |

Each template directory contains everything needed for one algorithm to work with ann-benchmarks framework.
