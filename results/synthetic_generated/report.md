# B1 运行报告
批次：`D:\dsh\idea\1017\b1-native-llm\examples\synthetic_json`

格式 json / 训练 120 / 测试 40 / 特征 128 / 类别 4

| 指标 | 结果 |
|---|---|
| 准确率（原始标签） | 40/40 = 100.00% |
| 准确率（同义词合并后） | 40/40 = 100.00% |
| 宏平均 F1 | 1.000000 |
| 无效输出 | 0 |
| 生成耗时 | 101.5 s |
| 沙箱执行 | 3.4 s |

## 一致性检查

- order_invariance: {'ok': True, 'changed': 0}
- label_sensitivity: {'ok': True, 'changed': 40}