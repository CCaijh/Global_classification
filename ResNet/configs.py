"""Shared defaults for all six experiments. Edit the two paths before running."""

train_configs = {
    "num_classes": 7,
    "batch_size": 120,
    "epochs": 100,
    "lr": 1e-4,
    "max_lr": 3e-4,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "pct_start": 0.3,
    "input_path": r"D:\cjh\dataset\Global_dataset\global_dataset_split_new",
    "output_path": r"D:\cjh\Benchmark model\logs\paper_exp\no_early_fixed",
    # Physical GPU indices used when clicking Run in PyCharm (no CLI arguments).
    # First wave: all three ResNet-50 runs + ResNet-101 seed42, one per GPU.
    # Second wave: start the last two only after GPU 0 / GPU 1 becomes free.
    "gpu_map": {
        "resnet50": {42: "0", 2026: "1", 3407: "2"},
        "resnet101": {42: "3", 2026: "0", 3407: "1"},
    },
    # At most 100 epochs; all six launchers share the same early-stopping rule.
    # Set this to 0 only if you explicitly want to disable early stopping.
    "early_stop_patience": 15,
    "early_stop_min_delta": 1e-4,
    "num_workers": 8,
    # Restore pinned batches used by the original early-stopping scripts.
    "pin_memory": True,
    # PyCharm progress stays live without forcing a redraw for every batch.
    "progress_mininterval": 1.0,
}
