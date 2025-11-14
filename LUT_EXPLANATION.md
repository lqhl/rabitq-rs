# LUT (Lookup Table) 详解

## 1. LUT 是什么？

**LUT** = **Lookup Table（查找表）**，是 FastScan 算法的核心优化技术。

### 1.1 基本原理

在 RaBitQ 量化中，每个向量被压缩为：
- **二进制码（binary code）**: 1-bit 符号位（正/负）
- **扩展码（extended code）**: 6-bit 幅值细化（for rabitq_bits=7）

计算查询向量 `q` 和数据向量 `v` 的二进制码距离需要：
```rust
distance = sum(q[i] * sign(v[i])) for i in 0..dim
```

### 1.2 LUT 的作用

**问题**: 直接计算太慢，需要 `dim` 次浮点乘法

**解决方案**: 预计算查找表！

FastScan 将维度分组为 4-bit 块（每 4 维一组）：
- 每个 4-bit 块有 2^4 = 16 种可能的二进制模式
- **预计算** 查询向量与这 16 种模式的点积
- 搜索时**查表**即可，无需重复计算

### 1.3 LUT 的结构

```rust
// src/fastscan.rs:11-21
pub struct QueryLut {
    pub lut_i8: Vec<i8>,      // 量化后的查找表 (i8 节省内存)
    pub delta: f32,           // 量化步长
    pub sum_vl_lut: f32,      // 偏移量总和
}
```

**维度**:
- 对于 960 维向量：960 / 4 = 240 个 codebook
- 每个 codebook 16 个条目（2^4 种组合）
- **总共**: 240 × 16 = **3840 个 LUT 条目**

**存储**:
- 原始：3840 × 4 bytes (f32) = 15,360 bytes
- 量化后：3840 × 1 byte (i8) = **3,840 bytes**

### 1.4 LUT 构建过程

```rust
// src/fastscan.rs:26-72
pub fn new(query: &[f32], padded_dim: usize) -> Self {
    let table_length = padded_dim * 4; // 每 4 维一组，共 16 种组合

    // Step 1: 生成浮点 LUT
    let mut lut_float = vec![0.0f32; table_length];
    simd::pack_lut_f32(query, &mut lut_float);

    // pack_lut_f32 的核心逻辑 (src/simd.rs:795-817):
    for i in 0..num_codebook {  // num_codebook = dim/4
        let q_offset = i * 4;
        let lut_offset = i * 16;

        lut[lut_offset] = 0.0;  // 模式 0000: 点积为 0
        for j in 1..16 {
            // 动态规划：利用前一个结果计算当前结果
            let prev_idx = j - lowbit(j);
            lut[lut_offset + j] = lut[lut_offset + prev_idx] + query[q_offset + KPOS[j]];
        }
    }

    // Step 2: 量化为 i8 (节省内存和加速 SIMD)
    let vl_lut = lut_float.min();
    let vr_lut = lut_float.max();
    let delta = (vr_lut - vl_lut) / 255.0;

    for i in 0..table_length {
        lut_i8[i] = ((lut_float[i] - vl_lut) / delta).round() as i8;
    }
}
```

**复杂度**:
- 时间：O(dim × 4) = O(3840) 浮点运算（for 960-dim）
- 空间：O(dim × 4) = 3,840 bytes (i8)

---

## 2. 为什么 MSTG 可能要缓存 LUT？

### 2.1 当前实现

**单次查询内已经复用了 LUT**：

```rust
// src/mstg/index.rs:159-181
pub fn search(&self, query: &[f32], params: &SearchParams) -> Vec<SearchResult> {
    // Step 3: 创建查询上下文（包含 LUT）
    let mut query_ctx = FastScanQueryContext::new(query.to_vec(), ex_bits);

    // 构建 LUT 一次（~15% 的查询时间）
    query_ctx.build_lut(padded_dim);  // ← 每次 search() 调用都要重建

    // Step 4: 搜索多个 posting lists，复用同一个 LUT
    let all_candidates: Vec<(u64, f32)> = selected_centroids
        .par_iter()
        .flat_map(|&cid| {
            let plist = &self.posting_lists[cid as usize];

            // 所有 posting list 共享同一个 query_ctx (包含 LUT)
            self.search_posting_list_fastscan(&query_ctx, plist, &plist.batch_data)
            //                                   ↑
            //                                 复用 LUT
        })
        .collect();
}
```

✅ **单次查询内的复用**: 已经实现！
❌ **多次查询间的复用**: 未实现！

### 2.2 缓存机会：批量查询场景

**场景**: 连续处理多个查询（例如批量推理、在线服务）

```rust
// 当前实现：每次都重建 LUT
for query in queries {
    index.search(&query, &params);  // ← 每次都调用 build_lut()
}

// 优化方案：缓存 LUT
let mut lut_cache = QueryContextCache::new();
for query in queries {
    index.search_with_cache(&query, &params, &mut lut_cache);
    //                                         ↑
    //                                  LUT 在多个查询间复用
}
```

### 2.3 缓存收益估算

**LUT 构建成本**（基于 GIST 960 维）:
- 占查询时间的 **~15%** (8-10ms / 59.7ms)
- 对于批量查询（100 个查询）:
  - 当前: 100 × 10ms = **1000ms** 浪费在重复构建 LUT
  - 缓存: 1 × 10ms = **10ms**（如果查询模式相似）
  - **节省**: ~990ms (**99% 加速** on LUT 构建部分)

**总体加速**:
- 单次查询加速：无（LUT 已经在 posting lists 间复用）
- 批量查询加速：**5-8ms per query**（15% × 59.7ms ≈ 9ms）
- **适用场景**: embedding 检索服务、批量推理

### 2.4 缓存限制

**何时不能缓存**？
- 不同维度的查询（LUT 大小不同）
- 不同的 rabitq_bits 配置（ex_bits 不同）
- LUT 会随查询向量变化，不是完全不变的

**缓存策略**：
```rust
// 简单缓存：假设同一批次的查询配置相同
if cache.is_valid_for(padded_dim, ex_bits) {
    // 复用已有的 LUT
} else {
    // 重建 LUT 并更新缓存
    cache.rebuild(query, padded_dim, ex_bits);
}
```

---

## 3. IVF 是怎么处理 LUT 的？

### 3.1 IVF 的 LUT 使用

**与 MSTG 完全相同的模式**：

```rust
// src/ivf.rs:1675-1771
pub fn search(&self, query: &[f32], params: &SearchParams) -> Result<Vec<SearchResult>> {
    // 预计算查询常量（包括旋转查询向量）
    let rotated_query = self.rotator.rotate(query);
    let mut query_precomp = QueryPrecomputed::new(rotated_query, self.ex_bits);

    // 构建 LUT 一次
    query_precomp.build_lut(self.padded_dim);  // ← 每次 search() 都重建

    // 搜索多个 clusters，复用同一个 LUT
    for &(cid, _) in cluster_scores.iter().take(nprobe) {
        let cluster = &self.clusters[cid];

        // 使用预计算的 query_precomp (包含 LUT)
        self.search_cluster_v2_batched(
            cid,
            cluster,
            &query_precomp,  // ← 复用 LUT
            g_add,
            g_error,
            dot_query_centroid,
            filter,
            &mut heap,
            params.top_k,
            &mut diagnostics,
        );
    }
}
```

### 3.2 IVF 单次查询内的 LUT 复用

```rust
// src/ivf.rs:1817-1822
fn search_cluster_v2_batched(
    &self,
    cluster: &ClusterData,
    query_precomp: &QueryPrecomputed,  // ← 包含预计算的 LUT
    ...
) {
    // 获取 LUT（已经在 search() 中构建好）
    let (lut_regular, lut_highacc) = if use_highacc {
        (None, query_precomp.lut_highacc.as_ref())
    } else {
        (query_precomp.lut.as_ref(), None)
    };

    // 使用 LUT 进行 FastScan 批量距离计算
    for batch_idx in 0..total_batches {
        // SIMD 累加使用 LUT
        simd::accumulate_batch_avx2(..., lut_regular, ...);
    }
}
```

### 3.3 IVF 是否缓存了 LUT？

**答案**: ❌ **没有**

- IVF 和 MSTG 采用**完全相同的策略**
- 单次查询内：✅ 复用 LUT（所有 clusters 共享）
- 多次查询间：❌ 不缓存（每次 `search()` 都重建）

**原因**:
1. **LUT 依赖查询向量**: 每个查询的 LUT 都不同
2. **内存开销**: 缓存 LUT 需要额外内存（~4KB per query for 960-dim）
3. **使用场景**: 单次查询优化优先，批量查询场景较少

---

## 4. MSTG vs IVF：LUT 使用对比

| 维度 | MSTG | IVF | 备注 |
|------|------|-----|------|
| **LUT 构建时机** | 每次 `search()` | 每次 `search()` | 相同 |
| **单次查询内复用** | ✅ 所有 posting lists 共享 | ✅ 所有 clusters 共享 | 相同 |
| **多次查询间缓存** | ❌ 不缓存 | ❌ 不缓存 | 相同 |
| **LUT 成本占比** | ~15% (9ms/59.7ms) | ~8% (3ms/36ms) | MSTG 略高 |
| **批量查询优化空间** | ⚠️ 可考虑缓存 | ⚠️ 可考虑缓存 | 两者都有机会 |

**关键差异**:
- **MSTG**: HNSW 导航成本高（20-25%），LUT 占比相对较小
- **IVF**: 线性扫描 clusters 成本低（5-10%），LUT 占比更小

---

## 5. 优化建议

### 5.1 何时考虑缓存 LUT？

**推荐场景**：
1. **批量查询**: 连续处理多个查询（>10 个）
2. **在线服务**: QPS > 100 的实时检索
3. **嵌入检索**: 文本/图像 embedding 批量搜索

**不推荐场景**：
1. 单次查询（无收益）
2. 低 QPS 离线分析（收益小）
3. 内存受限环境（缓存占用额外内存）

### 5.2 实现方案

```rust
// 方案 1: 简单缓存（假设同批次查询配置相同）
pub struct QueryContextCache {
    query_ctx: Option<QueryContext>,
    valid_for_dim: usize,
    valid_for_ex_bits: usize,
}

impl MstgIndex {
    pub fn search_batch(&self, queries: &[Vec<f32>], params: &SearchParams) -> Vec<Vec<SearchResult>> {
        let mut cache = QueryContextCache::new();

        queries.iter().map(|query| {
            if cache.is_valid(self.padded_dim, self.ex_bits) {
                // 复用缓存的 query_ctx，但更新查询向量
                cache.update_query(query);
            } else {
                // 重建 LUT
                cache.rebuild(query, self.padded_dim, self.ex_bits);
            }

            self.search_with_cache(query, params, &cache)
        }).collect()
    }
}
```

### 5.3 预期收益

**GIST 1M 批量查询 (100 queries)**:
- 当前: 100 × 59.7ms = **5,970ms**
- 缓存 LUT: 100 × (59.7 - 9)ms + 9ms = **5,079ms**
- **加速**: ~15% (**890ms 节省**)

**高 QPS 服务 (1000 QPS)**:
- 每秒节省: 1000 × 9ms = **9,000ms** ≈ **9 CPU cores** 的计算量
- **成本节省**: 显著降低服务器成本

---

## 6. 总结

### 6.1 LUT 本质
- **预计算表**: 将查询向量与所有 4-bit 二进制模式的点积预先计算
- **SIMD 优化**: 使用 i8 量化，配合 AVX2/AVX-512 指令加速
- **时间换空间**: 构建一次 O(dim×4)，查询时 O(1) 查表

### 6.2 当前复用策略
- ✅ **MSTG**: 单次查询内所有 posting lists 共享 LUT
- ✅ **IVF**: 单次查询内所有 clusters 共享 LUT
- ❌ **两者都未**: 跨查询缓存 LUT

### 6.3 优化价值
- **单次查询**: 无收益（已经在内部复用）
- **批量查询**: 高收益（5-8ms/query，~15% 加速）
- **优先级**: 中等（FastScan 已经是最大热点，LUT 只占 15%）

### 6.4 实现优先级排序

1. **高优先级** (立即可做):
   - ✅ 部分排序 top-k（2-3ms 加速，代码简单）

2. **中优先级** (批量查询场景):
   - ⚠️ LUT 缓存（5-8ms 加速，适合批量）
   - ⚠️ 并行 posting list 搜索（10-15ms 加速，多核）

3. **低优先级** (复杂优化):
   - ⚠️ HNSW 预取（2-4ms 加速，实现复杂）
   - ⚠️ 内存映射 posting lists（适合大索引）

---

**References**:
- LUT 构建: `src/fastscan.rs:26-72`, `src/simd.rs:795-817`
- MSTG LUT 使用: `src/mstg/index.rs:159-181`
- IVF LUT 使用: `src/ivf.rs:1675-1771`, `src/ivf.rs:1817-1822`
- FastScan 算法: C++ reference in `lut.hpp`, `estimator.hpp`
