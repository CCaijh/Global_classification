# ResNet 基准实验修正版

此版本使用一个训练入口和六个启动脚本。ResNet-50 / ResNet-101 各运行 42、2026、3407 三个种子。

## 运行前

1. 把整个文件夹解压到新目录，在原来能正常训练的 Python / PyTorch 环境中运行。
2. 修改 `configs.py` 的 `input_path` 和 `output_path`，或用 `--data-root`、`--output-root` 指定。Windows 路径使用 `r"..."`，路径有空格时命令行加双引号。
3. 数据根目录下必须有 `train`、`val`、`test1`、`test2`，各自包含相同的七个类别子目录。ImageFolder 按类别名称排序，程序会核对四个划分的类别映射。
4. 输入沿用原代码的预处理：仅 `ToTensor()`，不自动裁剪、resize、归一化或在线增强。使用你已经准备好的 448 × 448 图像；同一 batch 图像尺寸需一致。

## PyCharm 直接点击运行与四卡安排

六个 `start_*.py` 脚本可以直接在 PyCharm 点击运行，实时输出到 Run 窗口。无需额外填写显卡运行参数，`configs.py` 中的 `gpu_map` 自动指定物理显卡号。

| 启动脚本 | 默认物理显卡 | 启动安排 |
|---|---|---|
| `start_resnet50_seed42.py` | 0 | 第一批 |
| `start_resnet50_seed2026.py` | 1 | 第一批 |
| `start_resnet50_seed3407.py` | 2 | 第一批 |
| `start_resnet101_seed42.py` | 3 | 第一批 |
| `start_resnet101_seed2026.py` | 0 | 0号卡任务完全结束后 |
| `start_resnet101_seed3407.py` | 1 | 1号卡任务完全结束后 |

第一批最多同时运行四个脚本。后两组分别复用0号、1号卡，须等相应训练和最终测试完全结束后再启动。程序没有自动排队功能；不要一次点击启动全部六组。如果可用的卡号不是0～3，修改 `gpu_map` 的值即可，数字是物理编号。

输出路径只需在 `configs.py` 设置一个 `output_path`，所有实验在此路径下自动创建独立子目录，互不覆盖。

## 六个启动命令

也可以在此目录打开终端运行。不带 `--gpu` 时与 PyCharm 相同，读取 `gpu_map`。以下命令采用上表安排，第一批四组可在四个终端并行，后两组等待对应卡空闲。

```bash
python start_resnet50_seed42.py
python start_resnet50_seed2026.py
python start_resnet50_seed3407.py
python start_resnet101_seed42.py
python start_resnet101_seed2026.py
python start_resnet101_seed3407.py
```

六个启动脚本固定各自的模型和种子，不允许用 `--model` / `--seed` 改成另一组。

也可用 `--gpu 0` 等命令行参数覆盖 `gpu_map`，物理卡号以该参数为准。每个任务独立保存，互不覆盖。若 GPU 资源不足，不要同时在同一张卡启动多个 batch_size=120 的任务。

指定数据与输出路径示例：

```bash
python start_resnet50_seed42.py --gpu 1 --data-root "D:\cjh\dataset\Global_dataset\global_dataset_split_new" --output-root "D:\cjh\Benchmark model\logs\paper_exp\fixed_runs"
```

Linux 同样使用这些 Python 启动脚本；把路径换成你的 Linux 数据路径。

## 统一训练设置

| 项目 | 默认值 |
|---|---|
| Batch size | 120 |
| 最大训练轮数 | 100 |
| 早停 | patience=15，min_delta=1e-4，监测验证集 overall accuracy |
| SGD | lr=1e-4，momentum=0.9，Nesterov=True |
| 权重衰减 | 5e-4；一维参数与 bias 不衰减 |
| OneCycleLR | max_lr=3e-4，pct_start=0.3，cos |
| 模型选择 | 保存验证准确率每一次严格提高的权重；测试加载其中最好的一次 |
| 初始化 | 从头训练，未加载 ImageNet 预训练权重 |
| 网络结构 | 保留原文件中额外的 conv0 + BN + ReLU |

`min_delta=1e-4` 是准确率小数尺度，即 0.01 个百分点。早停改善量相对于上一次满足该阈值的准确率判断；达到100轮或连续15轮没有足够改善时结束。即使一次提高小于 min_delta，也会保存新的最佳权重；早停计数与最优权重保存分开判断。

OneCycle 的周期仍按最多100轮规划，与原代码一致；早停时可能尚未走完整个周期。OneCycle 会接管学习率，默认初始 LR 为 `max_lr/25 = 1.2e-5`，并非全程固定 `1e-4`；实际学习率写入日志和 `history.csv`。

六组实验应使用相同设置。需要统一改变设置时修改 `configs.py`。命令行也支持 `--batch-size`、`--epochs`、`--workers`、`--early-stop-patience` 等参数；汇总器会检查训练条件是否一致。

## 每次运行的输出

每次运行创建带模型、种子、UTC时间与随机标识的独立目录。目录内部包括：

| 文件 / 目录 | 内容 |
|---|---|
| `weights/*_best.pth` | 本次运行的最佳权重及模型、种子、run_id、类别映射等元数据 |
| `experiment_config.json` | 实际配置、数据数量与文件名/标签清单哈希、软件版本、源码哈希、最佳轮次、权重文件哈希和参数值哈希 |
| `class_indices.json` | 本次运行的类别索引，独立保存 |
| `train.log`、`history.csv` | 训练过程、学习率、验证准确率与 macro-F1 |
| `run_status.json` | running / complete / failed |
| `test1/`、`test2/` | 预测、混淆矩阵、每类指标与总体指标 |
| `final_test_summary.json` | 两个测试集的指标与实验身份 |

两个测试目录均导出 `predictions.csv`、`confusion_matrix.csv`、`class_metrics.csv`、`class_accuracy.csv`、`test_metrics.json`。指标包含真正的 macro-F1。`macro_accuracy` 为保留旧字段名而保存的平均召回率，不应代替 macro-F1。

加载权重时会核对模型、种子、run_id 与类别映射，防止测试时加载另一组实验。权重文件 SHA-256 用于追溯具体文件；参数值 SHA-256 不受种子或 run_id 元数据影响，可发现不同文件实际包含相同参数的情况。

新 `.pth` 文件包含 `state_dict` 和元数据。用于你其他推理代码时，需要这样加载：

```python
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
net.load_state_dict(checkpoint["state_dict"], strict=True)
net.eval()
```

本目录未包含旧版批量移动图片的预测脚本；不要直接把新权重传给只接受纯 state_dict 的旧脚本。

## 六组结束后汇总论文表格

```bash
python aggregate_results.py
```

也可指定输出根目录：

```bash
python aggregate_results.py --results-root "D:\cjh\Benchmark model\logs\paper_exp\fixed_runs"
```

生成 `benchmark_macro_f1_summary.csv` 和 `benchmark_macro_f1_summary_per_seed.csv`。macro-F1 以百分数表示，SD 为三个种子的样本标准差（ddof=1）。汇总器核对预测文件哈希、测试元数据、训练配置、数据文件名/标签清单与实现源码；相同预测或相同模型参数会打印警示。数据清单哈希不包含图像内容，不能代替图像内容重复/泄漏检查。

如果同一模型/种子有多个完整实验，程序会停止，避免静默选错；用 `--runs` 明确提供六个实验目录：

```bash
python aggregate_results.py --runs "run_dir_50_42" "run_dir_50_2026" "run_dir_50_3407" "run_dir_101_42" "run_dir_101_2026" "run_dir_101_3407"
```

## 修改范围与验证边界

性能修订：恢复部分原训练脚本使用的 `pin_memory=True`，GPU传输使用 `non_blocking=True`；非有限 loss 检查复用已有的标量读取，避免每个batch额外执行一次同步检查；进度条默认最多每秒刷新一次，并继续显示最新loss。训练、验证、保存权重与整轮耗时分别写入 `train.log`、`history.csv`；启动日志增加实际GPU型号、样本数量、batch数量与worker数量。网络、优化器、精度、早停规则与确定性设置保持原样，未添加 AMP 或更改 batch size。

这些修改用于减少开销和定位瓶颈，不能保证每轮恢复到20分钟。运行多组时，CPU、磁盘、内存与PCIe仍可能相互争用。若减速只在四组同时运行时出现，应先比较单组与并行时的batch速度。先确认比较的是同一实际模型、同一数据量、同一batch size和同一GPU型号。已有源码哈希校验会拒绝混用不同训练实现的结果，正式汇总需统一版本；不要为排查速度立即丢弃已有结果。

四个进程各设 `num_workers=8` 时，活跃训练DataLoader共最多32个worker；更多worker不保证更快。只有确认CPU/磁盘争用后，再统一测试更小的worker数。此版本没有打开persistent_workers，以免四任务同时驻留训练和验证worker导致内存压力。

已统一模型与种子定义、早停策略、权重与输出路径，添加测试身份验证及 macro-F1 导出，修复单样本 batch 的 `.squeeze()` 索引风险，并显式设置 DataLoader 生成器与 worker 随机种子。网络源码保持原样，优化器与 OneCycle 设置保持一致，未添加 AMP、EMA 或预训练权重。

交付前执行 Python 语法、六个启动入口、并发目录隔离、早停边界与 macro-F1 交叉验证。当前交付环境没有 PyTorch、torchvision 与 GPU 数据，未执行实际训练；首次使用需在你现有训练环境运行。
