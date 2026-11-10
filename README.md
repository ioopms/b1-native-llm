# 原生 LLM 测试管线：B1（一次代码生成）

一个可复现的评测管线，用标准化的方式回答一个问题：

> **同一个基座模型，在不给工具、不给执行反馈、只生成一次代码的条件下，能写出多好的分类程序？**

这就是文献里的 **B1 / direct prompting**：模型只拿到任务描述、数据 schema 和一份紧凑的训练集预览，
一次性输出一个完整程序；程序在沙箱里执行**一次**，不返回任何报错或分数；最后才用测试标签评分。

它和「让模型直接看数字报标签」（B0）是两个条件。B0 在光谱任务上会撞上上下文和数值比较的墙
（本仓库的动机实验：84 条参考 × 570 点原始数值塞进 prompt，准确率 50%，与多数类先验不可区分），
而 B1 把数值计算交给 CPU，测的是模型的**建模与工程能力**。

## 管线

```
batch/  (train + test)
   │   task.md + schema + 每类均值谱（64 bin 预览）
   ▼
LLM  ── 一次调用，temperature 0.2 ──►  candidate.py
   │
   ▼
sandbox  (只有 train/test，没有 ground_truth，禁网，超时)
   │   执行一次，不反馈
   ▼
predictions.json
   │
   ▼
本地评分（ground_truth） + 行序不变性 + 标签敏感性
```

对应 ScienceAgentBench 的 *Direct Prompting*：不与编程环境交互、给定输入一次生成程序。

## 快速开始

```bash
pip install -r requirements.txt

# 1) 离线自检：用内置参考程序跑通全流程，不调用 API
python examples/make_synthetic.py
python -m b1 selftest --batch examples/synthetic_json
python -m b1 selftest --batch examples/synthetic_csv

# 2) 已有候选程序，只执行 + 评分
python -m b1 run --batch examples/synthetic_json --candidate candidates/reference_json.py

# 3) 真实生成一次（PowerShell 设置 key；bash 用 export）
$env:deepseek_flash_api="sk-..."
python -m b1 generate --batch examples/synthetic_json
```

产物写到 `artifacts/<batch>/<mode>-<时间>/`：`prompt.txt`、`candidate.py`、`stdout.log`、
`stderr.log`、`predictions.json`、`report.md`、`report.json`。

> Python 环境需要 `numpy` 和 `pandas`。如果 `python -m b1` 报 `ModuleNotFoundError`，
> 说明当前解释器不是装了依赖的那个；换一个解释器或先 `pip install -r requirements.txt`。

## 批次格式

两种，自动识别：

- **JSON**（中小规模）：`reference_samples.json` + `samples.json` + `ground_truth.json`
- **CSV**（上万条光谱）：`train.csv` + `test.csv` + `ground_truth.json`

可选文件：`task.md`（任务描述，拼进 prompt 的 TASK 段）、`synonyms.json`（同义标签合并）。
字段细节见 [data/README.md](data/README.md)。

## 提示词

`b1/prompts.py` 只给四样东西：

1. `task.md` 里的任务描述（任务、介质、光谱类型）；
2. 输入文件的结构与形状；
3. 类别列表；
4. 一份**每类均值谱，降采样到 64 个 bin** 的训练集预览。

**不给方法建议**——用不用 SNV、LDA、PCA、kNN，全部由模型决定，这是 B1 要测的东西。
完整训练集不可能进上下文（拉曼数据是 2454 × 3276 ≈ 800 万个数字），所以用每类均值谱做预览；
这是“让模型看到训练集”的可行折中。

## 沙箱与检查

执行层（`b1/sandbox.py`）：

- 只把输入文件复制进临时目录，**`ground_truth.json` 不在里面**；
- `-E -s` 启动，`sitecustomize.py` 禁用 socket；默认 300 s 超时；
- 候选程序通过 `argv[1]` 读取数据目录，把 `predictions.json` 写到当前目录；
- 执行结束后才加载真实标签评分。

行为检查（`b1/checks.py`，默认开启）：

| 检查 | 做法 | 判据 |
|---|---|---|
| 行序不变性 | 打乱 train/test 的行顺序重跑 | 合法程序按 `sample_id` 应给出相同预测；`changed>0` = 依赖行序 |
| 标签敏感性 | 循环置换训练标签重跑 | 合法学习器的预测必须改变；`changed=0` = 硬编码/没在用标签 |
| 源码检查 | AST 扫描禁用导入与可疑字符串 | 只警告、不阻断（见“已知限制”） |

## 结果

| 数据集 | 代码来源 | 准确率 | 备注 |
|---|---|---|---|
| 橄榄油 MIR（4 类） | 内置参考实现 | 32/36 = 88.89% | 该划分的饱和上限；4 条样本 30 种分类器全错，3 条疑似标签错误 |
| 拉曼 API 化合物（32 标签） | 内置参考实现 | 97.54% / 合并同义词 100% | SNV + 收缩 LDA，约 5 s |
| 拉曼 API 化合物（32 标签） | **deepseek-flash 生成** | 98.39% / 合并 99.72% | 183.9 s、44,116 tokens 生成；6.5 s 执行；3 个错误全是 n-Heptane ↔ n-Hexane |
| 合成数据（4 类） | **deepseek-flash 生成** | 40/40 = 100% | 端到端自检 |

完整报告、生成的代码和 prompt 在 [results/](results/)，说明见 [results/README.md](results/README.md)。

几点解读：

- 拉曼数据集 32 个标签里只有 31 种化合物（`4-Methyl-2-pentanone` = `Methyl Isobutyl Ketone`），
  原始准确率会奖励“利用两批采集差异”，**合并后的指标才有化学意义**；
- 橄榄油数据集的 88.89% 是测试集上限，不是 B1 的能力上限；
- 这两个数据集都偏饱和，**不足以区分 B1 与智能体**；它们适合验证“模型能不能写出可运行的流程”。

## 目录结构

```
b1-native-llm/
├─ b1/                  # 管线：data / prompts / sandbox / checks / score / cli
├─ candidates/          # 内置参考程序（selftest 用，不是模型输出）
├─ examples/            # 合成数据与生成脚本
├─ tools/               # 数据集构建脚本（拉曼）
├─ data/                # 数据格式说明（不含原始数据）
├─ results/             # 归档的真实运行结果
├─ README.md
├─ requirements.txt
└─ .gitignore
```

## 已知限制

- **沙箱不是安全边界**。没有容器；候选程序仍可能通过硬编码绝对路径读到沙箱外的文件。
  对不可信代码请用 Docker/WSL，只读挂载数据、不挂 `ground_truth.json`。
- **源码检查只警告不阻断**。模型常用 `os.path.join`，`os` 会被标记但不会让运行失败。
- **近重复泄漏**。拉曼数据按化合物连续排列，每类多次扫描同一瓶样品；随机切分准确率会偏高。
- **饱和**。两个真实数据集都接近 100%，比较 scaffold 的差异需要更难的划分（掺假比例、
  跨仪器、每类只给 5 条参考等）。
- **API 成本**。`generate` 每次消耗一次调用；本仓库的示例结果约为 2 万–4.4 万 token/次。

## 复现

```bash
# 数据：见 data/README.md，把批次放到 data/ 下
python tools/build_raman_batch.py --csv path/to/raman.csv --out data/raman_b1
python -m b1 run --batch data/raman_b1 --candidate candidates/reference_csv.py
$env:deepseek_flash_api="sk-..."
python -m b1 generate --batch data/raman_b1
```

## 参考

- ScienceAgentBench: [arXiv:2410.05080](https://arxiv.org/abs/2410.05080)（direct prompting 的定义）
- ChemToolAgent / Tooling or Not Tooling: [NAACL 2025 Findings](https://doi.org/10.18653/v1/2025.findings-naacl.424)（原生 vs 工具增强的对照）
- SpecLLM: [Sci Rep 2026](https://doi.org/10.1038/s41598-026-75562-5)（光谱输入编码：原始数值 vs 紧凑文本）
- TabLLM: [arXiv:2210.10723](https://arxiv.org/abs/2210.10723)（表格序列化的 few-shot 基线）
- 拉曼数据集: [Scientific Data 2025](https://doi.org/10.1038/s41597-025-04848-6)

## 许可证

未指定。上传 GitHub 时请自行选择许可证（例如 MIT / CC BY 4.0）。
