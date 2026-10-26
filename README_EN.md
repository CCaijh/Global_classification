# Global classification

[简体中文](README.md) | **English**

## Project structure

- [AdaptiveNN](AdaptiveNN/README.md): AdaptiveNN models, training code, and usage instructions.
- [ResNet](ResNet/README.md): ResNet training and experiment code, with usage instructions.
- [weights](weights/README.md): Instructions for downloading, merging, and verifying model weights.
- [crop&resize.py](crop%26resize.py): Batch-crop images organized by year and resize them to a specified size.
- [part_random_noise_newest.py](part_random_noise_newest.py): Apply local noise augmentation and balance the number of samples per class.

Enter the relevant project directory and follow its README to install dependencies and run the program.

## Image preprocessing: cropping and resizing

[`crop&resize.py`](crop%26resize.py) uses Pillow to crop a specified region and then resize the image. By default, it resizes the `633×332`-pixel crop directly to `448×448`. Edit the following parameters in the script before running it. The current input and output paths are local paths and must be replaced with your own data paths.

| Parameter | Description | Current setting |
| --- | --- | --- |
| `source_dir` | Root directory of the original images, with subdirectories named by year | Set to your actual path |
| `target_dir` | Root directory for processed images | Set to your actual path |
| `years_to_process` | Names of the year directories to process | `2009`–`2013`, `2017`–`2019` |
| `crop_box` | Crop region in pixels, in `(left, top, right, bottom)` order | `(176, 199, 809, 531)` |
| `resize_size` | Output size in `(width, height)` order | `(448, 448)` |

Both input and output are organized by year, for example:

```text
source_dir/                 target_dir/
├── 2009/                   ├── 2009/
│   └── example.png         │   └── example.png
└── 2010/                   └── 2010/
    └── example.png             └── example.png
```

The script processes only images directly inside the selected year directories with the lowercase `.png` extension. The output preserves the year directories and original filenames. Make sure all selected year directories exist. Images with the same names in the output directories will be overwritten.

Install the dependency and run the script from the repository root. Keep the quotation marks because the filename contains `&`:

```sh
python -m pip install Pillow
python "crop&resize.py"
```

## Image augmentation: local noise and class balancing

[`part_random_noise_newest.py`](part_random_noise_newest.py) uses OpenCV, NumPy, scikit-image, and tqdm. Their package names are `opencv-python`, `numpy`, `scikit-image`, and `tqdm`, respectively. Input images must be organized into subdirectories by class. The output preserves the same class structure:

```text
input_path/                 output_path/
├── class_0/                ├── class_0/
│   └── example.png         │   ├── example.png
└── class_1/                │   └── example_gaussian_0.05_101.png
    └── example.png         └── class_1/
                                └── ...
```

Before running the script, edit `input_path`, `output_path`, and `image_count_threshold`. The default threshold is **30,000 images per class**. Each class directory must contain at least one image readable by OpenCV. The script does not filter files by extension.

The processing steps are:

1. If a class has at least the threshold number of images, randomly select that many original images and copy them to the output directory.
2. If a class has fewer images than the threshold, copy all original images first, then repeatedly generate augmented images from the originals until the count reaches the threshold.
3. For each augmented image, randomly choose a noise type and level. Apply noise first to `0`–`3` narrow vertical strips, then to `0`–`3` diagonal parallelogram regions. Each strip is `1`–`5` pixels wide. If the number of strips is `0`, at least one diagonal noise region is generated. The image dimensions remain unchanged.

Adjust the noise parameters in `noise_levels`:

| Noise type | `method` | Current levels and their meanings |
| --- | --- | --- |
| Gaussian noise | `gaussian` | `0.05`, `0.10`, `0.15`, passed as `var` to specify the variance |
| Poisson noise | `poisson` | `1`, `2`, `4`, used as multipliers of the Poisson noise result |
| Salt-and-pepper noise | `salt_pepper` | `0.1`, `0.15`, `0.2`, passed as `amount` to specify the proportion of noise |

Augmented images are saved as PNG files named `<original filename before its first dot>_<noise type>_<level>_<index>.png`. Use a new output directory where possible: the script copies original images with their existing filenames and does not remove augmented images from previous runs. The target count of 30,000 applies to the current run.

The Poisson noise branch currently uses `random_noise(..., seed=None)`. `seed` is a parameter from older versions of scikit-image. In newer environments that accept only `rng`, change it to `rng=None` or remove `seed=None`. See the [older parameter documentation](https://scikit-image.org/docs/0.21.x/api/skimage.util.html#skimage.util.random_noise) and the [current parameter documentation](https://scikit-image.org/docs/stable/api/skimage.util.html#skimage.util.random_noise).

After configuring the paths and installing the dependencies, run the following from the repository root:

```sh
python part_random_noise_newest.py
```

To crop images before augmenting them, run the preprocessing script first, organize its output into class-specific input directories, and then run the augmentation script. The two scripts run independently; the preprocessing script does not automatically organize images by class.
