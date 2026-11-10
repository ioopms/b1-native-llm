# Raman B1 运行报告
批次：raman_b1

训练 2454 / 测试 1056 / 特征 3276 / 类别 32

| 指标 | 结果 |
|---|---|
| 准确率（原始 32 标签） | 1039/1056 = 98.39% |
| 准确率（合并同义词后） | 1053/1056 = 99.72% |
| 运行时间 | 6.5 s |
| 错误数（合并后） | 3 |

## 错误样本

| 样本 | 真实 | 预测 |
|---|---|---|
| api_02783 | n-Heptane | n-Hexane |
| api_02898 | n-Hexane | n-Heptane |
| api_02901 | n-Hexane | n-Heptane |

## 一致性检查

- order_invariance: changed=0 ok=True
- label_sensitivity: changed=1056 ok=True