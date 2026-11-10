# B1 运行报告
批次：`C:\Users\czz\Downloads\四分类，橄榄 (2)`

格式 json / 训练 84 / 测试 36 / 特征 570 / 类别 4

| 指标 | 结果 |
|---|---|
| 准确率（原始标签） | 32/36 = 88.89% |
| 准确率（同义词合并后） | 32/36 = 88.89% |
| 宏平均 F1 | 0.919872 |
| 无效输出 | 0 |
| 沙箱执行 | 0.2 s |

## 一致性检查

- order_invariance: {'ok': True, 'changed': 0}
- label_sensitivity: {'ok': True, 'changed': 36}

## 错误样本

| 样本 | 真实 | 预测 |
|---|---|---|
| test_033 | Spain | Italy |
| test_034 | Spain | Italy |
| test_035 | Spain | Italy |
| test_036 | Spain | Italy |