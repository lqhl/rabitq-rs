# GIST 1M Benchmark Results - MSTG vs IVF

## Performance Comparison at Same Recall Level

| Recall@100 | MSTG QPS | MSTG Latency | IVF QPS | IVF Latency | MSTG Speedup |
|------------|----------|--------------|---------|-------------|--------------|
| 70%        | 65       | 15.4ms       | 9       | 105.8ms     | **6.9x**     |
| 80%        | 51       | 19.6ms       | 5       | 204.1ms     | **10.4x**    |
| 90%        | 25       | 39.8ms       | 5       | 204.1ms     | **5.1x**     |
| 95%        | 17       | 59.7ms       | 3       | 399.3ms     | **6.7x**     |

## Key Results

- **MSTG achieves 95.4% recall**, matching IVF's performance
- **At 95% recall**: MSTG is **6.7x faster** than IVF (17 vs 3 QPS)
- **Recommended configuration**: ef=800, epsilon=3.0 for 95% recall at 17 QPS

## MSTG Recall vs Parameters

| ef_search | epsilon | Recall Range | QPS Range |
|-----------|---------|--------------|-----------|
| 50        | 0.3-3.0 | 67-72%       | 98-160    |
| 100       | 0.3-3.0 | 67-90%       | 65-159    |
| 200       | 0.3-3.0 | 68-93%       | 38-155    |
| 400       | 0.3-3.0 | 67-91%       | 28-153    |
| **800**   | **0.8-3.0** | **89-95%** | **17-27** |
| 1600      | 0.8-3.0 | 90-95%       | 14-25     |
| 3200      | 0.8-3.0 | 90-95%       | 14-25     |

**Observation**: ef=800 provides the best recall/speed tradeoff. Higher ef_search values (1600/3200) provide minimal recall improvement.

## Files

- `recall_qps_fixed.csv` - Full parameter sweep results (50 configurations)
  - 42 MSTG configurations (7 ef_search × 6 epsilon values)
  - 8 IVF configurations (different nprobe values)

## Dataset

- **Name**: GIST 1M
- **Base vectors**: 1,000,000
- **Query vectors**: 1,000
- **Dimensions**: 960
- **Metric**: L2 distance

## Usage

Run the parameter sweep:
```bash
cargo run --release --example recall_qps_sweep
```

Generate plots (requires pandas/matplotlib):
```bash
cd benchmarks
python plot_recall_qps.py gist_1m_results/recall_qps_fixed.csv
```
