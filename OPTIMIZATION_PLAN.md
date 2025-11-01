# RaBitQ Rust 优化计划

## 基准性能
- 当前 QPS: 11,525 (小数据集测试)
- 目标 QPS: 2000-5000 (GIST 全数据集)
- Recall 基准: 0.511 (必须保持 > 0.5)

## 优化阶段

### Phase 1: LUT 量化优化 (最安全)
**目标**: 减少内存带宽压力，提升缓存利用率

**实现步骤**:
1. 将 LUT 从 `Vec<f32>` (4 bytes/entry) 转换为 `Vec<u8>` (1 byte/entry)
2. 量化公式:
   ```rust
   let scale = 255.0 / (max_val - min_val);
   let quantized = ((val - min_val) * scale).round() as u8;
   ```
3. 搜索时反量化:
   ```rust
   let dequantized = (quantized as f32) / scale + min_val;
   ```

**文件修改**: `src/ivf.rs` 的 `build_lut` 函数

**预期收益**:
- 内存带宽减少 75%
- L1/L2 缓存命中率提升

**风险评估**: 极低 - 仅影响精度 < 0.4%

---

### Phase 2: 内存预取优化
**目标**: 隐藏内存访问延迟

**实现步骤**:
1. 添加预取函数:
   ```rust
   #[inline(always)]
   fn prefetch_l1(ptr: *const u8) {
       #[cfg(target_arch = "x86_64")]
       unsafe {
           use std::arch::x86_64::_mm_prefetch;
           _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
       }
   }
   ```

2. 在批处理循环前预取下一批数据:
   ```rust
   // 预取下一批 32 个向量的数据
   for i in 0..20 {  // 20 cache lines ≈ 1280 bytes
       prefetch_l1(next_batch_ptr.add(i * 64));
   }
   ```

**文件修改**: `src/simd.rs` 和 `src/ivf.rs`

**预期收益**:
- 隐藏 200-600 CPU 周期的内存延迟
- 对顺序访问模式效果显著

**风险评估**: 极低 - 仅是提示，不影响正确性

---

### Phase 3: 64字节内存对齐
**目标**: 优化 SIMD 加载性能，减少跨缓存行访问

**实现步骤**:
1. 为批数据结构添加对齐属性:
   ```rust
   #[repr(C, align(64))]
   struct BatchData {
       bin_codes: Vec<u8>,
       f_add: [f32; 32],
       f_rescale: [f32; 32],
       f_error: [f32; 32],
   }
   ```

2. 使用 aligned allocation:
   ```rust
   let layout = Layout::from_size_align(size, 64).unwrap();
   let ptr = unsafe { alloc(layout) };
   ```

**文件修改**: `src/ivf.rs` 中的数据结构定义

**预期收益**:
- SIMD 指令可使用更快的 aligned load
- 避免跨缓存行的惩罚

**风险评估**: 低 - 仅影响内存布局

---

### Phase 4: 验证并优化累加器使用
**目标**: 确保充分利用指令级并行

**实现步骤**:
1. 验证当前代码是否正确使用 4 个独立累加器
2. 确保累加器之间无依赖:
   ```rust
   // Good: 独立累加
   accu0 = _mm256_add_epi16(accu0, res0);
   accu1 = _mm256_add_epi16(accu1, res1);
   accu2 = _mm256_add_epi16(accu2, res2);
   accu3 = _mm256_add_epi16(accu3, res3);

   // Bad: 有依赖链
   accu = _mm256_add_epi16(accu, res0);
   accu = _mm256_add_epi16(accu, res1);  // 等待上一条完成
   ```

**文件修改**: `src/simd.rs` 的 `accumulate_batch_avx2_impl`

**预期收益**:
- CPU 可同时执行多条指令
- 减少流水线停顿

**风险评估**: 低 - 仅验证现有实现

---

### Phase 5: Rayon 并行化
**目标**: 利用多核处理集群量化

**实现步骤**:
1. 添加 rayon 依赖:
   ```toml
   [dependencies]
   rayon = "1.7"
   ```

2. 并行化集群处理:
   ```rust
   use rayon::prelude::*;

   clusters.par_iter_mut()
       .for_each(|cluster| {
           // 量化集群中的向量
           quantize_cluster(cluster);
       });
   ```

**文件修改**: `src/ivf.rs` 的 `train` 函数

**预期收益**:
- 在多核系统上接近线性加速
- 特别适合大数据集训练

**风险评估**: 低 - 仅影响训练阶段，不影响搜索

---

## 测试协议

每个优化阶段后必须：

1. **运行 recall 测试**:
   ```bash
   python test_recall_accuracy.py
   ```
   确保 Average Recall > 0.5

2. **运行性能测试**:
   ```bash
   python test_ivf_optimized.py
   ```
   记录 QPS 变化

3. **运行完整基准测试** (如果 recall 正常):
   ```bash
   cargo run --release --bin ivf_rabitq -- \
       --base data/gist/gist_base.fvecs \
       --nlist 4096 \
       --bits 7 \
       --queries data/gist/gist_query.fvecs \
       --gt data/gist/gist_groundtruth.ivecs \
       --nprobe 64 \
       --max-base 10000 \
       --max-queries 100
   ```

4. **版本控制**:
   - 每个优化创建单独的 commit
   - 包含性能数据和 recall 验证结果

## 成功标准

- ✅ Recall 保持 > 0.5
- ✅ QPS 提升 > 20%
- ✅ 无内存泄漏或段错误
- ✅ 所有单元测试通过

## 风险缓解

如果任何优化导致 recall 下降:
1. 立即回滚该优化
2. 分析原因并记录到 CLAUDE.md
3. 尝试更保守的版本或跳过该优化

## 时间估算

- Phase 1 (LUT 量化): 2 小时
- Phase 2 (预取): 1 小时
- Phase 3 (对齐): 1 小时
- Phase 4 (验证累加器): 30 分钟
- Phase 5 (并行化): 2 小时

总计: ~6.5 小时