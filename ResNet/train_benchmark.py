"""One training/evaluation implementation for ResNet-50/101 and any seed."""

import argparse
import contextlib
import csv
import datetime as dt
import logging
import math
import os
from pathlib import Path
import random
import sys
import time

from configs import train_configs
from experiment_utils import (
    BestModelTracker, create_run_dir, dataset_index_sha256, metrics_from_confusion, save_metric_tables,
    sha256_file, state_dict_sha256, write_json,
)


def parse_args(argv=None, fixed_model=None, fixed_seed=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if fixed_model is None:
        parser.add_argument("--model", choices=["resnet50", "resnet101"], required=True)
    else:
        parser.set_defaults(model=fixed_model)
    if fixed_seed is None:
        parser.add_argument("--seed", type=int, required=True)
    else:
        parser.set_defaults(seed=fixed_seed)
    parser.add_argument("--data-root", default=train_configs["input_path"])
    parser.add_argument("--output-root", default=train_configs["output_path"])
    # Physical GPU index, with CUDA_VISIBLE_DEVICES set before importing torch.
    parser.add_argument("--gpu", default=None, help="Override the physical GPU from configs.py gpu_map")
    parser.add_argument("--cpu", action="store_true")
    for flag, key, value_type in [
        ("--batch-size", "batch_size", int), ("--epochs", "epochs", int),
        ("--lr", "lr", float), ("--max-lr", "max_lr", float),
        ("--momentum", "momentum", float), ("--weight-decay", "weight_decay", float),
        ("--pct-start", "pct_start", float), ("--workers", "num_workers", int),
        ("--early-stop-patience", "early_stop_patience", int),
        ("--early-stop-min-delta", "early_stop_min_delta", float),
    ]:
        parser.add_argument(flag, dest=key, type=value_type, default=train_configs[key])
    args = parser.parse_args(argv)
    if args.gpu is None:
        args.gpu = str(train_configs.get("gpu_map", {}).get(args.model, {}).get(args.seed, "0"))
    if args.batch_size < 1 or args.epochs < 1 or args.num_workers < 0:
        parser.error("batch-size/epochs must be positive; workers must be nonnegative")
    if args.early_stop_patience < 0 or args.early_stop_min_delta < 0:
        parser.error("early-stop-patience/min-delta must be nonnegative")
    if args.lr <= 0 or args.max_lr <= 0 or not 0 < args.pct_start < 1:
        parser.error("lr/max-lr must be positive; pct-start must be in (0, 1)")
    if not 0 < args.momentum < 1 or args.weight_decay < 0:
        parser.error("momentum must be in (0, 1); weight-decay must be nonnegative")
    if not 0 <= args.seed < 2 ** 32:
        parser.error("seed must be in [0, 2**32)")
    if not str(args.gpu).isdigit():
        parser.error("gpu must be one nonnegative physical GPU index")
    return args


def seed_worker(worker_id):
    # Module-level function so Windows DataLoader spawn can pickle it.
    import numpy as np
    import torch
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def seed_everything(seed, torch, np):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_logger(run_dir):
    logger = logging.getLogger(run_dir.name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    for handler in [logging.FileHandler(run_dir / "train.log", encoding="utf-8"),
                    logging.StreamHandler(sys.stdout)]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def parameter_groups(net, decay):
    groups = [{"params": [], "weight_decay": decay},
              {"params": [], "weight_decay": 0.0}]
    for name, param in net.named_parameters():
        if param.requires_grad:
            no_decay = param.ndim == 1 or name.endswith(".bias")
            groups[int(no_decay)]["params"].append(param)
    return groups


def make_loader(dataset, args, torch, seed_offset, shuffle):
    generator = torch.Generator()
    generator.manual_seed((args.seed + seed_offset) % (2 ** 32))
    return torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=min(os.cpu_count() or 1, args.num_workers) if args.batch_size > 1 else 0,
        worker_init_fn=seed_worker, generator=generator, drop_last=False,
        pin_memory=bool(train_configs.get("pin_memory", True)) and not args.cpu,
    )


def evaluate(net, loader, device, torch, tqdm, save_dir=None, experiment=None):
    classes = loader.dataset.classes
    count = len(classes)
    matrix = [[0] * count for _ in range(count)]
    offset = 0
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=False)
    handle_context = (save_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") if save_dir else contextlib.nullcontext(None)
    net.eval()
    with handle_context as handle, torch.no_grad():
        writer = None
        if handle is not None:
            fields = ["image_path", "true_label_id", "true_label_name", "pred_label_id", "pred_label_name", "correct"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
        for images, labels in tqdm(loader, desc="Test" if save_dir else "Validation",
                                   mininterval=train_configs.get("progress_mininterval", 1.0)):
            outputs = net(images.to(device, non_blocking=device.type == "cuda"))
            predicted = outputs.argmax(dim=1).detach().cpu().tolist()
            true_labels = labels.tolist()
            for i, (true, pred) in enumerate(zip(true_labels, predicted)):
                matrix[true][pred] += 1
                if writer is not None:
                    path, expected_label = loader.dataset.samples[offset + i]
                    if expected_label != true:
                        raise RuntimeError("Evaluation sample order mismatch")
                    writer.writerow({
                        "image_path": path, "true_label_id": true,
                        "true_label_name": classes[true], "pred_label_id": pred,
                        "pred_label_name": classes[pred], "correct": int(true == pred),
                    })
            offset += len(true_labels)
    if offset != len(loader.dataset):
        raise RuntimeError("Evaluation did not cover the entire dataset")
    metrics = metrics_from_confusion(matrix, classes)
    if save_dir:
        metrics.update({"split": save_dir.name, "experiment": experiment,
                        "predictions_sha256": sha256_file(save_dir / "predictions.csv")})
        save_metric_tables(save_dir, matrix, classes, metrics)
        write_json(save_dir / "test_metrics.json", metrics)
    return metrics


def main(argv=None, fixed_model=None, fixed_seed=None):
    args = parse_args(argv, fixed_model, fixed_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.cpu else str(args.gpu)
    try:
        import numpy as np
        import torch
        import torchvision
        from torchvision import datasets, transforms
        from tqdm import tqdm
        from model_resnet_448 import resnet50, resnet101
    except ImportError as exc:
        raise SystemExit("Missing dependency. Use your PyTorch training environment; see requirements.txt.") from exc

    seed_everything(args.seed, torch, np)
    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        raise SystemExit("Requested GPU is unavailable. Check --gpu or use --cpu explicitly.")
    data_root = Path(args.data_root).expanduser().resolve()
    # Preserve original preprocessing: ToTensor only, no resize/normalization/augmentation.
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_map = {}
    for split in ("train", "val", "test1", "test2"):
        root = data_root / split
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {root}")
        dataset_map[split] = datasets.ImageFolder(str(root), transform=transform)
    train_dataset = dataset_map["train"]
    if len(train_dataset.classes) != train_configs["num_classes"]:
        raise ValueError(f"Expected {train_configs['num_classes']} classes, found {train_dataset.classes}")
    for split, dataset in dataset_map.items():
        if dataset.class_to_idx != train_dataset.class_to_idx:
            raise ValueError(f"Train/{split} class mapping mismatch")

    run_dir = create_run_dir(args.output_root, args.model, args.seed)
    logger = make_logger(run_dir)
    checkpoint_path = run_dir / "weights" / f"{args.model}_seed{args.seed}_best.pth"
    checkpoint_path.parent.mkdir()
    config = {
        **vars(args), "data_root": str(data_root), "num_classes": len(train_dataset.classes),
        "architecture": "original model_resnet_448.py (conv0 retained)",
        "preprocessing": "ToTensor only", "optimizer": "SGD (nesterov=True)",
        "scheduler": "OneCycleLR (cos; div_factor=25; final_div_factor=10000)",
        "checkpoint_selection": "validation overall_accuracy",
        "pin_memory": bool(train_configs.get("pin_memory", True)) and device.type == "cuda",
        "non_blocking_cuda_transfer": device.type == "cuda",
        "progress_mininterval": train_configs.get("progress_mininterval", 1.0),
    }
    experiment = {
        "run_id": run_dir.name, "model": args.model, "seed": args.seed,
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint_path), "config": config,
        "class_to_idx": train_dataset.class_to_idx,
        "dataset_sizes": {split: len(ds) for split, ds in dataset_map.items()},
        "dataset_index_sha256": {split: dataset_index_sha256(ds) for split, ds in dataset_map.items()},
        "environment": {
            "python": sys.version, "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "device": str(device),
            "physical_gpu": None if args.cpu else str(args.gpu),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "source_sha256": {name: sha256_file(Path(__file__).parent / name)
                          for name in ("train_benchmark.py", "experiment_utils.py", "model_resnet_448.py", "configs.py")},
    }
    write_json(run_dir / "experiment_config.json", experiment)
    write_json(run_dir / "class_indices.json", {i: name for i, name in enumerate(train_dataset.classes)})
    write_json(run_dir / "run_status.json", {"status": "running", "run_id": run_dir.name})
    logger.info("Model=%s seed=%s GPU=%s device=%s", args.model, args.seed, args.gpu, device)
    logger.info("Independent run directory: %s", run_dir)
    logger.info("batch_size=%s epochs=%s early_stop_patience=%s", args.batch_size, args.epochs, args.early_stop_patience)
    logger.info("Data root=%s; dataset sizes=%s", data_root, experiment["dataset_sizes"])
    logger.info("GPU name=%s", experiment["environment"]["gpu_name"])
    started = time.monotonic()
    try:
        net = {"resnet50": resnet50, "resnet101": resnet101}[args.model](num_classes=len(train_dataset.classes)).to(device)
        train_loader = make_loader(train_dataset, args, torch, 0, True)
        val_loader = make_loader(dataset_map["val"], args, torch, 1, False)
        logger.info("Train batches=%d; workers=%d; pin_memory=%s; validation batches=%d",
                    len(train_loader), train_loader.num_workers, train_loader.pin_memory, len(val_loader))
        optimizer = torch.optim.SGD(parameter_groups(net, args.weight_decay), lr=args.lr,
                                    momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.max_lr, pct_start=args.pct_start,
            steps_per_epoch=len(train_loader), epochs=args.epochs, anneal_strategy="cos",
            div_factor=25.0, final_div_factor=10000.0,
        )
        logger.info("OneCycle initial LR=%g (scheduler controls LR)", optimizer.param_groups[0]["lr"])
        tracker = BestModelTracker(args.early_stop_patience, args.early_stop_min_delta)
        loss_function = torch.nn.CrossEntropyLoss()
        best_epoch = None
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            history = csv.DictWriter(handle, fieldnames=[
                "epoch", "train_loss", "val_accuracy", "val_macro_f1", "lr", "best_epoch",
                "train_seconds", "val_seconds", "checkpoint_seconds", "epoch_seconds", "train_images_per_second",
            ])
            history.writeheader()
            for epoch in range(1, args.epochs + 1):
                epoch_started = time.perf_counter()
                net.train()
                loss_sum = 0.0
                seen = 0
                progress = tqdm(train_loader, desc=f"{args.model} seed{args.seed} epoch {epoch}/{args.epochs}",
                                mininterval=train_configs.get("progress_mininterval", 1.0))
                for images, labels in progress:
                    optimizer.zero_grad()
                    images = images.to(device, non_blocking=device.type == "cuda")
                    labels = labels.to(device, non_blocking=device.type == "cuda")
                    outputs = net(images)
                    loss = loss_function(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    loss_value = loss.detach().item()
                    # Reuse the existing scalar read; avoid a second GPU sync before backward.
                    if not math.isfinite(loss_value):
                        raise RuntimeError(f"Non-finite loss at epoch {epoch}")
                    seen += labels.size(0)
                    loss_sum += loss_value * labels.size(0)
                    progress.set_postfix(loss=f"{loss_value:.6f}", refresh=False)
                train_seconds = time.perf_counter() - epoch_started
                val_started = time.perf_counter()
                val_metrics = evaluate(net, val_loader, device, torch, tqdm)
                val_seconds = time.perf_counter() - val_started
                improved, stop = tracker.update(val_metrics["overall_accuracy"])
                checkpoint_seconds = 0.0
                if improved:
                    checkpoint_started = time.perf_counter()
                    best_epoch = epoch
                    temporary = checkpoint_path.with_suffix(".pth.tmp")
                    torch.save({
                        "state_dict": net.state_dict(), "model": args.model, "seed": args.seed,
                        "run_id": run_dir.name, "epoch": epoch,
                        "val_accuracy": val_metrics["overall_accuracy"],
                        "val_macro_f1": val_metrics["macro_f1"],
                        "class_to_idx": train_dataset.class_to_idx, "config": config,
                    }, temporary)
                    temporary.replace(checkpoint_path)
                    checkpoint_seconds = time.perf_counter() - checkpoint_started
                row = {"epoch": epoch, "train_loss": loss_sum / seen,
                       "val_accuracy": val_metrics["overall_accuracy"], "val_macro_f1": val_metrics["macro_f1"],
                       "lr": scheduler.get_last_lr()[0], "best_epoch": best_epoch,
                       "train_seconds": train_seconds, "val_seconds": val_seconds,
                       "checkpoint_seconds": checkpoint_seconds,
                       "epoch_seconds": time.perf_counter() - epoch_started,
                       "train_images_per_second": seen / train_seconds if train_seconds else 0.0}
                history.writerow(row)
                handle.flush()
                logger.info("Epoch %d train_loss=%.6f val_acc=%.6f val_macro_F1=%.6f best_epoch=%s",
                            epoch, row["train_loss"], row["val_accuracy"], row["val_macro_f1"], best_epoch)
                logger.info("Timing epoch=%d: train=%.1fs validation=%.1fs checkpoint=%.1fs total=%.1fs throughput=%.2f images/s",
                            epoch, train_seconds, val_seconds, checkpoint_seconds, row["epoch_seconds"], row["train_images_per_second"])
                if stop:
                    logger.info("Early stopping at epoch %d", epoch)
                    break
        # Load ONLY this run's checkpoint and verify its identity before evaluation.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for key, expected in [("model", args.model), ("seed", args.seed), ("run_id", run_dir.name),
                              ("class_to_idx", train_dataset.class_to_idx)]:
            if checkpoint.get(key) != expected:
                raise RuntimeError(f"Checkpoint identity mismatch: {key}")
        checkpoint_sha = sha256_file(checkpoint_path)
        weights_sha = state_dict_sha256(checkpoint["state_dict"])
        net.load_state_dict(checkpoint["state_dict"], strict=True)
        del checkpoint
        experiment.update({"best_epoch": best_epoch, "best_val_accuracy": tracker.best_accuracy,
                           "checkpoint_sha256": checkpoint_sha, "state_dict_sha256": weights_sha,
                           "epochs_completed": epoch})
        write_json(run_dir / "experiment_config.json", experiment)
        logger.info("Loaded own best checkpoint: %s", checkpoint_path)
        logger.info("Checkpoint SHA256=%s; tensor SHA256=%s", checkpoint_sha, weights_sha)
        summary = {"status": "complete", "experiment": experiment}
        for offset, split in enumerate(("test1", "test2"), start=2):
            loader = make_loader(dataset_map[split], args, torch, offset, False)
            summary[split] = evaluate(net, loader, device, torch, tqdm, run_dir / split, experiment)
            logger.info("%s macro-F1=%.6f%% accuracy=%.6f%%", split,
                        summary[split]["macro_f1_percent"], 100 * summary[split]["overall_accuracy"])
        write_json(run_dir / "final_test_summary.json", summary)
        write_json(run_dir / "run_status.json", {"status": "complete", "run_id": run_dir.name,
                                                  "elapsed_seconds": time.monotonic() - started})
        logger.info("Finished. Results: %s", run_dir)
    except BaseException as exc:
        write_json(run_dir / "run_status.json", {"status": "failed", "run_id": run_dir.name,
                                                  "error": str(exc), "elapsed_seconds": time.monotonic() - started})
        logger.exception("Experiment failed")
        raise
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
    return run_dir


def launch(model, seed):
    return main(fixed_model=model, fixed_seed=seed)


if __name__ == "__main__":
    main()
