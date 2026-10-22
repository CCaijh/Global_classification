# 模型权重

完整权重存放在 [GitHub Releases：weights-2026-10-22](https://github.com/CCaijh/Global_classification/releases/tag/weights-2026-10-22)。此目录保存下载清单、SHA256 校验值和还原脚本。

## 下载文件

从上面的 Release 下载以下 8 个文件，并放到同一个目录（例如本目录）。两个较大的权重各拆为 3 个二进制分卷，每个分卷不超过 1920 MiB；分卷需要合并后才能作为 `.pth` 权重使用。

| 原始权重 | 下载文件 |
| --- | --- |
| `checkpoint-best.pth` | [part001](https://github.com/CCaijh/Global_classification/releases/download/weights-2026-10-22/checkpoint-best.pth.part001)、[part002](https://github.com/CCaijh/Global_classification/releases/download/weights-2026-10-22/checkpoint-best.pth.part002)、[part003](https://github.com/CCaijh/Global_classification/releases/download/weights-2026-10-22/checkpoint-best.pth.part003) |
| `checkpoint-best_macro.pth` | [part001](https://github.com/CCaijh/Global_classification/releases/download/weights-2026-10-22/checkpoint-best_macro.pth.part001)、[part002](https://github.com/CCaijh/Global_classification/releases/download/weights-2026-10-22/checkpoint-best_macro.pth.part002)、[part003](https://github.com/CCaijh/Global_classification/releases/download/weights-2026-10-22/checkpoint-best_macro.pth.part003) |
| `resnet101_seed2026_best.pth` | [完整文件](https://github.com/CCaijh/Global_classification/releases/download/weights-2026-10-22/resnet101_seed2026_best.pth) |
| `resnet50_seed2026_best.pth` | [完整文件](https://github.com/CCaijh/Global_classification/releases/download/weights-2026-10-22/resnet50_seed2026_best.pth) |

校验清单也可从 Release 下载：[manifest.json](https://github.com/CCaijh/Global_classification/releases/download/weights-2026-10-22/manifest.json)、[SHA256SUMS.txt](https://github.com/CCaijh/Global_classification/releases/download/weights-2026-10-22/SHA256SUMS.txt)、[FILE_SHA256SUMS.txt](https://github.com/CCaijh/Global_classification/releases/download/weights-2026-10-22/FILE_SHA256SUMS.txt)。仓库此目录中保留相同的清单。

## 校验和还原

安装 Python 3 后，在仓库根目录运行：

```sh
python weights/restore_weights.py --assets weights --output weights
```

如果下载文件放在其他位置，将 `--assets` 改为该目录；`--output` 指定还原后的权重保存目录。脚本仅使用 Python 标准库，先检查分卷大小与 SHA256，再按顺序合并，并检查完整文件的大小与 SHA256。两个未分卷的 ResNet 权重也会校验并复制到输出目录。

已存在且校验正确的文件会跳过；如同名文件内容不同，脚本会停止，不会覆盖。还原过程中需要足够的磁盘空间同时保存下载文件和完整权重，建议至少预留 18 GiB 空间。

`SHA256SUMS.txt` 对应 8 个下载文件，`FILE_SHA256SUMS.txt` 对应 4 个完整 `.pth` 文件。Linux 可在下载目录运行 `sha256sum -c SHA256SUMS.txt`，macOS 可运行 `shasum -a 256 -c SHA256SUMS.txt` 进行分卷校验。

`adaptivenn/`、`resnet/` 保留原有空目录结构；训练和评估时请依据各项目 README 指定实际权重路径。
