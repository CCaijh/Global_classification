# Global classification

**简体中文** | [English](README_EN.md)

## 项目目录

- [AdaptiveNN](AdaptiveNN/README.md)：AdaptiveNN 模型、训练代码及使用说明。
- [ResNet](ResNet/README.md)：ResNet 训练与实验代码及使用说明。
- [weights](weights/README.md)：模型权重的下载、合并和校验说明。
- [crop_ad_resize.py](crop_ad_resize.py)：批量裁剪图片、填充两侧并对齐背景，统一缩放为 `448×448`。
- [part_random_noise_newest.py](part_random_noise_newest.py)：按类别进行局部噪声增强和样本数量均衡。

请进入对应项目目录，并按照各自的 README 安装依赖和运行程序。

## 图片预处理：裁剪、填充与缩放

[`crop_ad_resize.py`](crop_ad_resize.py) 使用 OpenCV 和 NumPy 对图片依次进行裁剪、两侧填充、背景颜色对齐和缩放。运行前先修改脚本顶部的 `path`（输入目录）和 `outpath`（输出目录）；当前路径是本机路径，需要替换为自己的数据路径。

当前处理设置如下，坐标和尺寸单位均为像素：

| 设置 | 当前值及说明 |
| --- | --- |
| 输入目录 `path` | 直接存放待处理图片的目录 |
| 输出目录 `outpath` | 保存处理结果，目录不存在时自动创建 |
| 裁剪切片 | `img_array[200:531, 177:807, :]`，即左 `177`、上 `200`、右 `807`、下 `531`，右、下边界不包含在内 |
| 裁剪尺寸 | 宽 `630` × 高 `331` |
| 两侧填充 | 左右各 `39` 列，填充后为宽 `708` × 高 `331` |
| 背景颜色 | OpenCV 的 BGR 顺序为 `(172, 177, 187)` |
| 行均值阈值 | `160 < meanOfRow[i] < 205`，满足条件的行在两侧填充对应背景色 |
| 输出尺寸 | `(448, 448)`，使用 `cv2.INTER_CUBIC` 插值 |

处理流程如下：

1. 使用 `cv2.IMREAD_UNCHANGED` 读取原图并裁剪指定区域，将裁剪图底行设置为背景色。
2. 建立白色画布，左右各扩展 `39` 列，贴入裁剪图，并设置部分边界像素的背景色。
3. 计算画布每行的像素均值，在满足上述阈值的行中，将两侧填充区域设置为背景色。
4. 将填充后的图片缩放为 `448×448`，保留原文件名保存。

脚本每次处理一个目录，输入和输出结构示例：

```text
path/                       outpath/
├── example_01.png          ├── example_01.png
└── example_02.png          └── example_02.png
```

脚本仅处理输入目录下一层、扩展名为小写 `.png` 的图片。若数据按年份组织，请将 `path`、`outpath` 分别设为对应年份目录，并逐年运行。请使用有效的输入目录，输出目录中的同名图片会被覆盖。

建议输入为 8 位三通道彩色图片，至少宽 `807`、高 `531`，以匹配当前固定裁剪区域。裁剪、填充及边界位置写在源码中，修改时需要同步调整相关索引，包括右侧填充范围 `669:708`。

当前文件导入了 OpenCV、NumPy、scikit-image 和 Pillow，对应依赖可通过以下命令安装。在仓库根目录运行：

```sh
python -m pip install opencv-python numpy scikit-image Pillow
python crop_ad_resize.py
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
