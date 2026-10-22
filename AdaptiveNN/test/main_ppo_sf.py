import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'




# 1. 全量日志重定向：确保所有打印内容（混淆矩阵等）都记入 output.log
class Logger(object):
    def __init__(self, filename="output.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


sys.path.insert(0, './models')
from pathlib import Path
from timm.data.mixup import Mixup
from timm.models import create_model
from optim_factory import create_optimizer, LayerDecayValueAssigner
from dataset_utils_sf import build_dataset
from engine_ppo_sf import train_one_epoch, evaluate
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import models.dynamic_deitS_sf


# 2. Focal Loss 定义：解决类别预测难易度不一的问题
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if inputs.size() == targets.size():  # 适配 Mixup 后的 SoftTarget
            ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)

        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


def get_args_parser():
    parser = argparse.ArgumentParser('AdaptiveNN training for custom dataset', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--update_freq', default=1, type=int)

    # 模型与数据路径参数
    parser.add_argument('--model', default='dynamic_deitS', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--nb_classes', default=4, type=int)
    parser.add_argument('--data_path', default='D:\cjh\dataset\\2002_2015_class4_split\\train', type=str, help='训练集图像根目录 (2002-2015/train)')
    parser.add_argument('--eval_data_path', default='D:\cjh\dataset\\2002_2015_class4_split\\val', type=str, help='验证集图像根目录 (2002-2015/val)')
    parser.add_argument('--test_data_path', default='D:\cjh\dataset\HA2016-SpreadF-img-4class-448', type=str, help='测试集图像根目录 (HA2016)')
    parser.add_argument('--manual_feat_path', default='D:\cjh\dataset\\2002_2015_class4_split\\feature_txt', type=str, help='特征 CSV 根目录 (feature_txt)')
    parser.add_argument('--data_set', default='image_folder', choices=['image_folder', 'IMNET'])

    # Focal Loss 超参数（将保存在 args.json）
    parser.add_argument('--focal_gamma', default=2.0, type=float)
    parser.add_argument('--focal_alpha', default=[1.0, 1.5, 2.0, 1.0], type=float, nargs='+',
                        help='类别权重：第3类(index 2)设为2.0，第2类(index 1)设为1.5')

    # 系统与保存
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--lr', type=float, default=4e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)

    # 分布式补齐参数（适配 utils.py 报错修复）
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true', default=False)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--distributed', action='store_true', default=False)

    # AdaptiveNN RL 相关
    parser.add_argument('--seq_l', default=4, type=int)
    parser.add_argument('--ppo_std_start', type=float, default=0.25)
    parser.add_argument('--ppo_std_end', type=float, default=0.10)
    parser.add_argument('--update_policy_freq', default=1, type=int)

    # 补齐 dataset_utils 需要的参数
    parser.add_argument('--imagenet_default_mean_and_std', action='store_true', default=True)
    parser.add_argument('--train_interpolation', default='bicubic', type=str)
    parser.add_argument('--color_jitter', type=float, default=0.4)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1')
    parser.add_argument('--reprob', type=float, default=0.25)
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--resplit', action='store_true')
    parser.add_argument('--crop_pct', type=float, default=0.875)

    return parser


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # 3. 时间戳文件夹与参数保存
    if utils.is_main_process():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(args.output_dir, timestamp)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # 开启日志记录
        sys.stdout = Logger(os.path.join(args.output_dir, "output.log"))

        # 保存所有超参数（含 Focal Loss 参数）
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
        print(f"Experimental folder created: {args.output_dir}")

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # 4. 数据加载逻辑：训练、验证、测试分离
    # 训练集：图像在 data_path，特征在 feature_txt/train
    dataset_train, args.nb_classes = build_dataset(
        args.data_path,
        os.path.join(args.manual_feat_path, 'train'),
        True,
        args
    )

    # 验证集：图像在 eval_data_path，特征在 feature_txt/train
    dataset_val, _ = build_dataset(
        args.eval_data_path,
        os.path.join(args.manual_feat_path, 'train'),
        False,
        args
    )

    # 测试集：图像在 test_data_path，特征在 feature_txt/test
    dataset_test, _ = build_dataset(
        args.test_data_path,
        os.path.join(args.manual_feat_path, 'test'),
        False,
        args
    )

    sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=True)
    sampler_val = torch.utils.data.DistributedSampler(dataset_val, shuffle=False)
    sampler_test = torch.utils.data.DistributedSampler(dataset_test, shuffle=False)

    data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train, batch_size=args.batch_size,
                                                    num_workers=args.num_workers, pin_memory=True, drop_last=True)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val, batch_size=args.batch_size,
                                                  num_workers=args.num_workers, pin_memory=True, drop_last=False)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, sampler=sampler_test, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # 5. 模型初始化
    model = create_model(args.model, seq_len=args.seq_l, num_classes=args.nb_classes, feature_in_chans=384,
                         glance_input_size=112, focus_patch_size=112, remaining_blocks=4, recover_n=5, pretrained=False)
    model.to(device)

    # 6. 优化器与 Focal Loss 配置
    optimizer = create_optimizer(args, model, bone_lr_scale=1.0)
    loss_scaler = NativeScaler()

    class_weights = torch.tensor(args.focal_alpha).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
    print(f"Loss: FocalLoss | Alpha: {args.focal_alpha} | Gamma: {args.focal_gamma}")

    utils.auto_load_model(args=args, model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

    # 6. 训练、验证与实时测试循环
    max_mean_acc = 0.0
    start_time = time.time()
    num_steps = len(dataset_train) // (args.batch_size * utils.get_world_size())
    lr_schedule = utils.cosine_scheduler(args.lr, 1e-6, args.epochs, num_steps)

    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler,
                                      args=args, lr_schedule_values=lr_schedule, num_training_steps_per_epoch=num_steps,
                                      update_freq=1)

        print(f"\n--- Epoch {epoch} 验证集评估 (用于保存最佳权重) ---")
        val_stats = evaluate(data_loader_val, model, device, args=args)

        # 核心：根据验证集“平均分类准确率”判断
        current_mean_acc = val_stats.get('mean_class_acc', 0)

        if current_mean_acc > max_mean_acc:
            max_mean_acc = current_mean_acc
            if utils.is_main_process():
                utils.save_model(args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                                 loss_scaler=loss_scaler, epoch="best")
                print(f"*** 验证集平均准确率新高: {max_mean_acc:.2f}%. 正在同步对 HA2016 测试集进行测试...")

                # 实时大考：发现最佳模型即对测试集跑一遍
                evaluate(data_loader_test, model, device, args=args)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()}, 'epoch': epoch}
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    print(f'Training completed in {str(datetime.timedelta(seconds=int(time.time() - start_time)))}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AdaptiveNN Training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)