# Global classification

## 项目目录

- [AdaptiveNN](AdaptiveNN/README.md)：AdaptiveNN 模型、训练代码及使用说明。
- [ResNet](ResNet/README.md)：ResNet 训练与实验代码及使用说明。
- [weights](weights/README.md)：模型权重的下载、合并和校验说明。
- [crop&resize.py](crop%26resize.py)：按年份批量裁剪图片，并统一调整为指定尺寸。
- [part_random_noise_newest.py](part_random_noise_newest.py)：按类别进行局部噪声增强和样本数量均衡。

请进入对应项目目录，并按照各自的 README 安装依赖和运行程序。

## 图片预处理：裁剪与缩放

[`crop&resize.py`](crop%26resize.py) 使用 Pillow 处理图片，先裁剪指定区域，再调整尺寸。默认将 `633×332` 像素的裁剪区域直接缩放为 `448×448`。运行前修改脚本中的以下参数；当前输入、输出路径是本机路径，需要替换为自己的数据路径。

| 参数 | 说明 | 当前设置 |
| --- | --- | --- |
| `source_dir` | 原始图片根目录，下面按年份建立子目录 | 按实际路径修改 |
| `target_dir` | 处理结果根目录 | 按实际路径修改 |
| `years_to_process` | 要处理的年份目录名 | `2009`—`2013`、`2017`—`2019` |
| `crop_box` | 裁剪区域，顺序为 `(左, 上, 右, 下)`，单位为像素 | `(176, 199, 809, 531)` |
| `resize_size` | 输出尺寸，顺序为 `(宽, 高)` | `(448, 448)` |

输入和输出均按年份组织，例如：

```text
source_dir/                 target_dir/
├── 2009/                   ├── 2009/
│   └── example.png         │   └── example.png
└── 2010/                   └── 2010/
    └── example.png             └── example.png
```

脚本仅处理所选年份目录下一层、扩展名为小写 `.png` 的图片，输出保留年份目录和原文件名。请确保所选年份目录均已存在，输出目录中的同名图片会被覆盖。

安装依赖后，在仓库根目录运行；文件名包含 `&`，命令中需要保留引号：

```sh
python -m pip install Pillow
python "crop&resize.py"
```

## 图片增强：局部噪声与类别均衡

[`part_random_noise_newest.py`](part_random_noise_newest.py) 使用 OpenCV、NumPy、scikit-image 和 tqdm，依赖包名分别为 `opencv-python`、`numpy`、`scikit-image`、`tqdm`。输入数据需要按类别建立子目录，输出保留相同的类别结构：

```text
input_path/                 output_path/
├── class_0/                ├── class_0/
│   └── example.png         │   ├── example.png
└── class_1/                │   └── example_gaussian_0.05_101.png
    └── example.png         └── class_1/
                                └── ...
```

运行前修改脚本中的 `input_path`、`output_path` 和 `image_count_threshold`，其中数量阈值默认是 **每类 30,000 张**。每个类别目录应包含可由 OpenCV 读取的图片，并至少有一张图片；脚本不按扩展名过滤文件。

处理流程如下：

1. 类别图片数达到或超过阈值时，随机选取阈值数量的原图复制到输出目录。
2. 类别图片数不足时，先复制全部原图，再循环从原始图片生成增强图片，直到计数达到阈值。
3. 每张增强图片随机选择一种噪声和一个级别，先对 `0`—`3` 条竖向窄条区域加噪，再对 `0`—`3` 个斜向平行四边形区域加噪。窄条宽度为 `1`—`5` 像素；当窄条数量为 `0` 时，至少生成一个斜向噪声区域。图片尺寸保持不变。

噪声参数可在 `noise_levels` 中调整：

| 噪声类型 | `method` | 当前级别及含义 |
| --- | --- | --- |
| 高斯噪声 | `gaussian` | `0.05`、`0.10`、`0.15`，传入 `var`，表示方差 |
| 泊松噪声 | `poisson` | `1`、`2`、`4`，表示泊松噪声结果的乘数 |
| 椒盐噪声 | `salt_pepper` | `0.1`、`0.15`、`0.2`，传入 `amount`，表示噪声比例 |

增强图片保存为 PNG，命名格式为 `<原文件名首个点号前的部分>_<噪声类型>_<级别>_<序号>.png`。建议使用新的输出目录，因为脚本会复制同名原图，且不会清理以前生成的增强图片；30,000 的计数目标以本次处理为准。

当前脚本的泊松噪声分支使用 `random_noise(..., seed=None)`。`seed` 是旧版 scikit-image 参数；在仅接受 `rng` 参数的新版环境中，需要将该处改为 `rng=None` 或移除 `seed=None`，参见 [旧版参数说明](https://scikit-image.org/docs/0.21.x/api/skimage.util.html#skimage.util.random_noise)和[当前参数说明](https://scikit-image.org/docs/stable/api/skimage.util.html#skimage.util.random_noise)。

配置路径和依赖后，在仓库根目录运行：

```sh
python part_random_noise_newest.py
```

如果先裁剪再增强，请先运行预处理脚本，然后将结果整理为按类别划分的输入目录，再运行增强脚本。两个脚本独立运行，预处理脚本不会自动进行类别划分。
