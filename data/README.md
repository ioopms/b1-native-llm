# 数据

本仓库**不包含**任何原始光谱数据，只包含数据格式说明、构建脚本和小型合成示例。

## 支持的两种批次格式

### JSON（对象数组，适合中小规模）

```
batch/
├─ reference_samples.json   [{"sample_id", "spectrum": {"<波数>": 强度, ...}, "label"}]
├─ samples.json             [{"sample_id", "spectrum": {...}}]
├─ ground_truth.json        [{"sample_id", "label"}]
├─ task.md                  （可选）任务描述，会拼进 prompt 的 TASK 段
└─ synonyms.json            （可选）同义标签映射，用于合并后的指标
```

### CSV（宽表，适合上万条光谱）

```
batch/
├─ train.csv       sample_id, label, <特征列...>
├─ test.csv        sample_id, <特征列...>
├─ ground_truth.json
├─ task.md
└─ synonyms.json
```

## 两个真实数据集

### 1. 橄榄油中红外（4 类产地，JSON）

- 84 条训练参考、36 条测试，每条 570 个波数点（798.892–1896.8085 cm⁻¹）。
- 由实验平台的「原生 LLM 直接分类」批次导出，字段就是上面的 JSON 格式。
- 注意：测试集里有 3 条（test_034–036）标签为 Spain，但光谱最近邻全在 Italy，
  疑似标签错误；4 个方法族、30 种分类器都无法把它们分对。因此 B1 的 88.89% 是该划分的上限，
  不是方法能力上限。

### 2. 拉曼 API 化合物（32 标签，CSV）

- 公开数据：*Open-source Raman spectra of chemical compounds for active pharmaceutical
  ingredient development*, Scientific Data (2025), DOI [10.1038/s41597-025-04848-6](https://doi.org/10.1038/s41597-025-04848-6)。
- 3510 条光谱 × 3276 个波数点（150–3425 cm⁻¹，1 cm⁻¹），32 个标签。
- **32 个标签其实只有 31 种化合物**：`4-Methyl-2-pentanone` 与 `Methyl Isobutyl Ketone`
  是同一物质（MIBK）。`synonyms.json` 负责合并，报告里同时给出原始与合并后的准确率。
- 原始 CSV 按化合物连续排列，每类 100–145 条是同一瓶样品的多次扫描；随机切分存在近重复泄漏，
  准确率会偏高。这个数据集适合当“管线能不能跑”的健全性检查，不适合当能力基准。

构建批次：

```bash
python tools/build_raman_batch.py \
  --csv "path/to/raman_spectra_api_compounds.csv" \
  --out data/raman_b1
```

切分：按化合物分层 70/30，seed 42。

## 合成示例

`examples/synthetic_json` 与 `examples/synthetic_csv` 由 `examples/make_synthetic.py` 生成：
4 类 × (30 训练 + 10 测试) × 128 特征，确定性随机数。用于离线自检，可直接提交到仓库。
