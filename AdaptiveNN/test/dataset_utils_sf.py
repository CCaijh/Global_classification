import os
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD


class AdaptiveCombinedDataset(Dataset):
    """
    自定义数据集：同步加载图像和对应的 126 维人工特征。
    """

    def __init__(self, img_root, feat_root, transform=None):
        self.base_dataset = datasets.ImageFolder(img_root, transform=transform)
        self.transform = transform
        self.feature_dim = 125
        self.manual_features = {}

        print(f"正在从 {feat_root} 加载人工特征 CSV...")
        if os.path.exists(feat_root):
            for csv_file in os.listdir(feat_root):
                if csv_file.endswith('.csv'):
                    csv_path = os.path.join(feat_root, csv_file)
                    try:
                        df = pd.read_csv(csv_path)
                        for _, row in df.iterrows():
                            # 提取 image_id 并在字典中建立 126 维特征映射
                            img_id = str(row['image_id']).strip()
                            feat_values = row.iloc[1:127].values.astype('float32')
                            self.manual_features[img_id] = torch.tensor(feat_values)
                    except Exception as e:
                        print(f"解析 CSV {csv_path} 时出错: {e}")
        else:
            print(f"警告: 特征路径 {feat_root} 不存在")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        path, target = self.base_dataset.imgs[index]
        img = self.base_dataset.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        # 移除后缀进行 ID 匹配 (如 HA419...PNG -> HA419...)
        img_id = os.path.splitext(os.path.basename(path))[0]
        m_feat = self.manual_features.get(img_id, torch.zeros(self.feature_dim))

        return img, m_feat, target


def build_dataset(img_root, feat_root, is_train, args):
    """
    根据传入的路径构建数据集
    """
    transform = build_transform(is_train, args)

    # 初始化自定义数据集
    dataset = AdaptiveCombinedDataset(img_root, feat_root, transform=transform)

    # 自动获取类别数量
    nb_classes = len(dataset.base_dataset.classes)
    return dataset, nb_classes

def build_transform(is_train, args):
    """
    极简转换：仅 Resize，无数据增强，无裁剪。
    """
    mean = IMAGENET_DEFAULT_MEAN if args.imagenet_default_mean_and_std else IMAGENET_INCEPTION_MEAN
    std = IMAGENET_DEFAULT_STD if args.imagenet_default_mean_and_std else IMAGENET_INCEPTION_STD

    t = [
        # 直接缩放到输入尺寸，不进行任何裁剪或随机增强
        transforms.Resize((args.input_size, args.input_size), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
    return transforms.Compose(t)