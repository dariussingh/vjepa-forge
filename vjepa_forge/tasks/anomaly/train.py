from vjepa_forge.tasks.anomaly.runtime import train_from_runtime_config


class AnomalyTrainer:
    def __init__(self, model, **kwargs: object) -> None:
        self.model = model
        self.kwargs = dict(kwargs)

    def run(self):
        data_cfg = dict(self.model.data_cfg)
        if "data" in self.kwargs:
            data_cfg["_path"] = str(self.kwargs["data"])
        train_cfg = {
            "epochs": self.kwargs.get("epochs", data_cfg.get("epochs", 10)),
            "batch_size": self.kwargs.get("batch_size", data_cfg.get("batch_size", 1)),
            "num_workers": self.kwargs.get("num_workers", data_cfg.get("num_workers", 0)),
            "device": self.kwargs.get("device", data_cfg.get("device", "cpu")),
            "save": self.kwargs.get("save", data_cfg.get("save", True)),
            "save_period": self.kwargs.get("save_period", data_cfg.get("save_period", 0)),
            "resume": self.kwargs.get("resume", data_cfg.get("resume", False)),
            "project": self.kwargs.get("project", data_cfg.get("project")),
            "name": self.kwargs.get("name", data_cfg.get("name")),
            "exist_ok": self.kwargs.get("exist_ok", data_cfg.get("exist_ok", False)),
            "lr": data_cfg.get("lr", 1.0e-4),
            "weight_decay": data_cfg.get("weight_decay", 1.0e-4),
            "lr_mode": data_cfg.get("lr_mode", "manual"),
            "reference_batch_size": data_cfg.get("reference_batch_size", self.kwargs.get("batch_size", data_cfg.get("batch_size", 1))),
            "reference_lr": data_cfg.get("reference_lr", data_cfg.get("lr", 1.0e-4)),
            "lr_scale_rule": data_cfg.get("lr_scale_rule", "sqrt"),
            "seed": data_cfg.get("seed", 7),
        }
        val_cfg = {
            "batch_size": data_cfg.get("val_batch_size", self.kwargs.get("batch_size", 1)),
            "num_workers": data_cfg.get("val_num_workers", data_cfg.get("num_workers", 0)),
            "split": data_cfg.get("val_split", "val"),
            "threshold_std_multiplier": data_cfg.get("threshold_std_multiplier", 3.0),
            "smoothing_window": data_cfg.get("smoothing_window", 9),
            "checkpoint_target": data_cfg.get("checkpoint_target", "best"),
            "checkpoint_path": data_cfg.get("checkpoint_path"),
        }
        config = {
            "model": self.model.model_cfg,
            "data": data_cfg,
            "train": train_cfg,
            "val": val_cfg,
            "predict": {"split": "test"},
            "export": {},
            "output": data_cfg.get("output", {}),
        }
        return train_from_runtime_config(config)


__all__ = ["AnomalyTrainer"]
