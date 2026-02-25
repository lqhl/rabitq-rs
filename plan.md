# MSTG 优化与 Disk Mode 实现计划

## 背景

当前 MSTG 在 Cohere 100K-768d 上性能出色 (99.4% recall, 6.18x faster than IVF at 98%), 但 replication factor ~5x 导致内存占用约 124 MB, 是 IVF 的 ~5 倍. 此外 disk mode 的框架已搭建但不可用于实际搜索.

本计划三个目标:
1. 控制 replication factor 到 ~1.2x, 在低内存开销下保持尽可能高的 recall
2. 实现可用的 disk mode 搜索
3. 对 IVF, MSTG-memory, MSTG-disk 三者进行公平 benchmark

## 第 1 步: 控制 replication factor

### 1.1 分析 epsilon 与 replication 的关系

当前 `closure_epsilon=0.3`, threshold 为 `closest_dist_sq * (1+0.3)^2 = closest_dist_sq * 1.69`, 导致平均 ~5 个候选通过. 目标是 ~1.2x, 即 100K 向量 → ~120K 总条目, 平均每个向量分配到 1.2 个簇.

思路: 不改 assign 逻辑, 只调 epsilon 值. 理论上极小的 epsilon (如 0.02~0.05) 就能产生 ~1.2x 的 replication.

### 1.2 实现方案

在 benchmark 脚本中用多个 epsilon 值进行 sweep, 找到 replication ≈ 1.2 的 epsilon, 然后固定使用该值:

- epsilon sweep: 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.3
- 每个 epsilon 记录: replication factor, 构建时间, 内存占用
- 选择 replication factor 最接近 1.2 的 epsilon 值
- 用这个最优 epsilon 运行完整 recall/latency sweep

### 1.3 需要修改的代码

- `examples/bench_ivf_vs_mstg.rs`: 增加 epsilon sweep 逻辑, 或者增加 `--epsilon` 命令行参数
- 打印 replication factor 和内存占用以便比较

## 第 2 步: 实现 disk mode

### 2.1 当前问题

1. `load_from_path` 把所有 posting list 反序列化到内存 (InMemory 模式), 没有 mmap 加载路径
2. `PostingDataSource::Mmap` 的 `iter()` 返回空数组
3. `with_posting_list` 每次搜索都做 bincode 反序列化 + `build_batch_layout()`, 开销太大
4. search 路径已经使用 `with_posting_list()` 抽象, 理论上 InMemory 和 Mmap 透明

### 2.2 实现方案

#### 2.2.1 mmap 加载 API

在 `io.rs` 中添加 `MstgIndex::load_from_path_mmap(path)`:

- 加载 centroid index (HNSW) 到内存 (这部分本来就在 memory tier)
- 加载 config 和 PostingListDirectory 到内存 (metadata tier)
- 用 memmap2 mmap 整个 posting list 文件, 不反序列化 (disk tier)
- 返回 `PostingDataSource::Mmap(mmap, base_offset)`

#### 2.2.2 优化 with_posting_list 的反序列化开销

每次搜索时从 mmap 反序列化 posting list 是必要的 (这就是 disk mode 的本质: 用 CPU 换内存), 但 `build_batch_layout()` 的开销可以优化:

方案 A (简单): 接受反序列化 + build_batch_layout 的开销, 先让它跑起来, 测量实际性能损失
方案 B (优化): 把 batch_layout 预计算并序列化到磁盘, 加载时直接读取 (需要改 PostingList 的序列化格式)

先采用方案 A, 确保功能正确, 然后看 benchmark 结果决定是否需要方案 B.

#### 2.2.3 save 路径调整

当前 `save_main_index` 已经线性写入 posting list 并记录 offset, 应该已经兼容 mmap 读取. 需验证:

- disk_offset 是否正确指向每个 posting list 的起始位置
- 序列化格式是否兼容: 前 8 字节为长度, 后跟 bincode 数据

### 2.3 需要修改的文件

- `src/mstg/io.rs`: 添加 `load_from_path_mmap` 方法
- `src/mstg/index.rs`: 确保 `PostingDataSource::Mmap` 与搜索路径兼容
- `src/mstg/search.rs`: 确认 search 在 mmap 模式下正常工作 (理论上已兼容)

## 第 3 步: 三方 benchmark

### 3.1 测试配置

在 Cohere 100K-768d 数据集上, 对以下三种配置进行测试:

| 配置 | 说明 |
|------|------|
| IVF | 标准 IVF-RaBitQ, nlist=sqrt(N) |
| MSTG-mem | MSTG 内存模式, epsilon 调到 replication ~1.2x |
| MSTG-disk | MSTG disk 模式 (mmap), 同样的 epsilon 和索引 |

### 3.2 测量指标

每个配置测量:
- 构建时间 (ms)
- 内存占用 (MB) - 对 disk mode 只计 centroid index + directory
- Recall@100 sweep
- 延迟 (us/query)
- QPS

### 3.3 benchmark 脚本修改

`examples/bench_ivf_vs_mstg.rs` 需要扩展:

1. 增加 `--epsilon` 参数
2. 增加 MSTG disk mode 测试: build → save → load_mmap → search
3. 输出三方对比表

### 3.4 预期输出格式

```
========================================================
  IVF vs MSTG-mem vs MSTG-disk Benchmark
========================================================

  IVF: nlist=316, build=8000ms
  MSTG-mem: epsilon=0.03, replication=1.2, build=7000ms, mem=28MB
  MSTG-disk: same index, mmap load, mem=3MB (centroid only)

  --- IVF ---
  nprobe  recall  latency  QPS
  ...

  --- MSTG-mem ---
  profile  recall  latency  QPS
  ...

  --- MSTG-disk ---
  profile  recall  latency  QPS
  ...

  --- Head-to-head ---
  ...
```

## 执行顺序

1. 先实现第 2 步 (disk mode), 因为这是功能性代码修改
2. 然后修改 benchmark 脚本 (第 1 步 + 第 3 步)
3. 运行 benchmark, 收集结果
4. 根据结果决定是否需要优化 disk mode (方案 B)
