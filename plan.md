# v0.8.0 发布计划

## 1. 准备发布分支
- 从 `main` 分支创建新分支 `release/v0.8.0`

## 2. 自动化测试与性能验证
对 `ivf`, `mstg-mem`, `mstg-disk` 在 `dataset/` 下的数据集进行全面测试，并保存结果。

### 2.1 测试数据集
- `cohere_100k_768d`

### 2.2 测试维度
- **构建阶段**: 构建时间, 索引大小（磁盘/内存）
- **加载阶段**: 加载时间 (InMemory vs Mmap)
- **搜索阶段**: Recall@10, Recall@100, Latency (ms/query), QPS

### 2.3 执行脚本
使用 `examples/bench_ivf_vs_mstg.rs` (或相应命令行工具) 运行测试。

## 3. 文档同步与去标识
更新所有文档，确保现状与实现保持一致，正式移除 MSTG 的 "experimental" 标识。

### 3.1 README.md
- [ ] 移除所有 ⚠️ Experimental 警告。
- [ ] 更新版本号为 `0.8.0`。
- [ ] 更新 "When to Use MSTG vs IVF+RaBitQ" 表格，将 MSTG 标记为 Production Ready。
- [ ] 确保 MSTG 的使用示例准确无误。

### 3.2 GEMINI.md & 其他文档
- [ ] `GEMINI.md`: 去除 MSTG 的 experimental 标识。
- [ ] 检查并更新 `docs/*.md` 中的陈旧描述。

## 4. 版本发布准备
- [ ] 更新 `Cargo.toml` 中的版本号为 `0.8.0`。
- [ ] 更新 `pyproject.toml` 中的版本号。
- [ ] 运行全量单元测试 `cargo test`。
- [ ] 运行 Python 绑定测试 `make test-python`。
- [ ] 运行 `cargo clippy` 和 `cargo fmt` 确保代码质量。

## 5. 发布
- 将 `release/v0.8.0` 合并至 `main`。
- 打标签 `v0.8.0`。
