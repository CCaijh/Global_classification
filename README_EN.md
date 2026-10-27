# Global classification

[简体中文](README_CN.md) | **English**

## Project structure

- [AdaptiveNN](AdaptiveNN/README.md): AdaptiveNN models, training code, and usage instructions.
- [ResNet](ResNet/README.md): ResNet training and experiment code, with usage instructions.
- [weights](weights/README.md): Instructions for downloading, merging, and verifying model weights.
- [crop_ad_resize.py](crop_ad_resize.py): Batch-crop images, add side padding with background color alignment, and resize them to `448×448`.
- [part_random_noise_newest.py](part_random_noise_newest.py): Apply local noise augmentation and balance the number of samples per class.

Enter the relevant project directory and follow its README to install dependencies and run the program.

## Image preprocessing: cropping, padding, and resizing

[`crop_ad_resize.py`](crop_ad_resize.py) uses OpenCV and NumPy to crop images, add side padding, align the background color, and resize the result. Before running the script, edit `path` (the input directory) and `outpath` (the output directory) at the top of the source file. The current paths are local paths and must be replaced with your own data paths.

The current settings are listed below. All coordinates and dimensions are in pixels:

| Setting | Current value and description |
| --- | --- |
| Input directory `path` | Directory containing the images to process |
| Output directory `outpath` | Directory for processed images; created automatically if it does not exist |
| Crop slice | `img_array[200:531, 177:807, :]`: left `177`, top `200`, right `807`, bottom `531`; the right and bottom bounds are exclusive |
| Crop size | Width `630` × height `331` |
| Side padding | `39` columns on each side, producing width `708` × height `331` |
| Background color | `(172, 177, 187)` in OpenCV BGR order |
| Row mean threshold | `160 < meanOfRow[i] < 205`; the side padding in qualifying rows is filled with the background color |
| Output size | `(448, 448)`, using `cv2.INTER_CUBIC` interpolation |

The processing steps are:

1. Read the image with `cv2.IMREAD_UNCHANGED`, crop the specified region, and set the crop's bottom row to the background color.
2. Create a white canvas with `39` additional columns on each side, place the cropped image on it, and set designated boundary pixels to the background color.
3. Compute the mean across channels and columns for each row of the canvas, then fill both side strips with the background color in rows that meet the threshold above.
4. Resize the padded image to `448×448` and save it with its original filename.

The script processes one directory at a time. Example input and output structure:

```text
path/                       outpath/
├── example_01.png          ├── example_01.png
└── example_02.png          └── example_02.png
```

The script processes only images directly inside the input directory with the lowercase `.png` extension. If your data is organized by year, set `path` and `outpath` to the corresponding year directories and run the script once for each year. Use a valid input directory. Images with the same names in the output directory will be overwritten.

For the fixed crop, 8-bit, three-channel color images at least `807` pixels wide and `531` pixels high are recommended. The crop, padding, and boundary positions are hard-coded in the source. When changing them, update the related indices together, including the right padding range `669:708`. These settings are not command-line options.

The script imports OpenCV, NumPy, scikit-image, and Pillow. Install all four packages, even though the current processing code does not use the scikit-image or Pillow imports. Run the following from the repository root:

```sh
python -m pip install opencv-python numpy scikit-image Pillow
python crop_ad_resize.py
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
