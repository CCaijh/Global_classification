import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix  # 新增：用于计算混淆矩阵

import utils


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None,
                    args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('ppo_std_this_iter', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    for focus_step_index in range(args.seq_l):
        metric_logger.add_meter(f'pos_x_mean_{focus_step_index}', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter(f'pos_y_mean_{focus_step_index}', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter(f'pos_x_std_{focus_step_index}', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter(f'pos_y_std_{focus_step_index}', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    mse_loss = torch.nn.MSELoss()
    optimizer.zero_grad()

    # 修改：解包 samples, manual_feats, targets
    for data_iter_step, (samples, manual_feats, targets) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):

        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step

        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if epoch < param_group['fix_step']:
                    param_group["lr"] = 0.
                elif lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        if args.ppo_std_decay == 'cosine':
            if epoch < args.warmup_epochs:
                ppo_std_this_iter = args.ppo_std_start
            else:
                ppo_std_this_iter = args.ppo_std_end + (args.ppo_std_start - args.ppo_std_end
                                                        ) * lr_schedule_values[it] / args.lr
        elif args.ppo_std_decay == 'linear':
            ppo_std_this_iter = args.ppo_std_end + (args.ppo_std_start - args.ppo_std_end
                                                    ) * (1 - it / len(lr_schedule_values))
        else:
            raise NotImplementedError

        samples = samples.to(device, non_blocking=True)
        manual_feats = manual_feats.to(device, non_blocking=True)  # 新增：特征转到设备
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            # 修改：传入 manual_feats
            expected_outputs = model(samples, manual_feats=manual_feats, seq_l=args.seq_l,
                                     ppo_std_this_iter=ppo_std_this_iter)

            outputs_reg_focus_net = expected_outputs['outputs_reg_focus_net']
            loss_reg_focus_net = criterion(outputs_reg_focus_net, targets)
            loss_reg_focus_net = args.loss_reg_focus_net_weight * loss_reg_focus_net

            out_teacher = expected_outputs['x_focus'][-1].detach()

            loss_KD = F.kl_div(
                F.log_softmax(expected_outputs['x_glance'][-1] / args.kd_temp, dim=1),
                F.softmax(out_teacher / args.kd_temp, dim=1),
                reduction='batchmean'
            ) * (args.kd_temp ** 2)

            loss_KD = loss_KD + sum(
                [
                    F.kl_div(
                        F.log_softmax(_x_focus / args.kd_temp, dim=1),
                        F.softmax(out_teacher / args.kd_temp, dim=1),
                        reduction='batchmean'
                    ) * (args.kd_temp ** 2)
                    for _x_focus in expected_outputs['x_focus'][:-1]
                ]
            )

            loss_glance = criterion(expected_outputs['x_glance'][-1], targets)
            loss_focus = sum(
                [criterion(_x_focus, targets) for _x_focus in expected_outputs['x_focus']]
            )
            # 这里的 loss 会基于拼接了人工特征后的输出进行反向传播
            loss = loss_reg_focus_net + loss_focus + loss_glance + loss_KD * args.kd_alpha
            loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        loss /= update_freq
        grad_norm = loss_scaler(loss, optimizer,
                                model=model, skip_backbones=False, skip_policy_net=True,
                                clip_grad=max_norm,
                                parameters=model.parameters(),
                                update_grad=(data_iter_step + 1) % update_freq == 0)

        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()

        # PPO 逻辑保持原始实现，由于奖励计算基于 detach 后的分类 loss，融合特征会自动影响策略学习 [cite: 14, 253]
        if data_iter_step == 0:
            ppo_update_collect_dict = []
            targets_collect_dict = []

        ppo_update_collect_dict.append(expected_outputs)
        targets_collect_dict.append(targets)

        if (data_iter_step + 1) % args.update_policy_freq == 0:
            # ... (PPO 更新逻辑保持不变，详见原始代码 155-218 行) ...
            pass

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        # ... (指标更新逻辑) ...

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args=None):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    all_preds, all_targets = [], []
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, 'Eval:'):
        images, manual_feats, target = batch
        images, manual_feats, target = images.to(device), manual_feats.to(device), target.to(device)

        with torch.cuda.amp.autocast():
            expected_outputs = model(images, manual_feats=manual_feats, seq_l=args.seq_l)
            output = expected_outputs['x_focus'][-1]
            loss = criterion(output, target)

        acc1, _ = accuracy(output, target, topk=(1, 2))
        _, preds = torch.max(output, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=images.shape[0])

    metric_logger.synchronize_between_processes()

    # 计算混淆矩阵及各类别准确率
    cm = confusion_matrix(all_targets, all_preds)
    class_accs = []
    class_names = data_loader.dataset.base_dataset.classes

    if utils.is_main_process():
        print("\n" + "=" * 30 + " 分类报告 " + "=" * 30)
        for i, name in enumerate(class_names):
            acc = (cm[i, i] / cm[i, :].sum() * 100) if cm[i, :].sum() > 0 else 0
            class_accs.append(acc)
            print(f"  - {name.ljust(15)} Accuracy: {acc:.2f}%")

        # 计算 4 个分类的平均准确率
        mean_class_acc = np.mean(class_accs)
        print(f"\n* 平均分类准确率 (Mean Class Acc): {mean_class_acc:.3f}%")
        print("=" * 69 + "\n")
    else:
        # 非主进程也需要计算用于返回，但无需打印
        for i in range(len(class_names)):
            acc = (cm[i, i] / cm[i, :].sum() * 100) if cm[i, :].sum() > 0 else 0
            class_accs.append(acc)
        mean_class_acc = np.mean(class_accs)

    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    results['mean_class_acc'] = mean_class_acc  # 关键：返回平均准确率
    return results