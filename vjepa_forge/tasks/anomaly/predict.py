from vjepa_forge.tasks.anomaly.runtime import predict_from_runtime_config


class AnomalyPredictor:
    split = "test"

    def __init__(self, model, **kwargs: object) -> None:
        self.model = model
        self.kwargs = dict(kwargs)

    def run(self):
        data_cfg = dict(self.model.data_cfg)
        if "data" in self.kwargs:
            data_cfg["_path"] = str(self.kwargs["data"])
        config = {
            "model": self.model.model_cfg,
            "data": data_cfg,
            "train": {"device": self.kwargs.get("device", data_cfg.get("device", "cpu"))},
            "val": {
                "batch_size": data_cfg.get("batch_size", 1),
                "num_workers": data_cfg.get("num_workers", 0),
                "split": data_cfg.get("val_split", "val"),
                "threshold_std_multiplier": data_cfg.get("threshold_std_multiplier", 3.0),
                "smoothing_window": data_cfg.get("smoothing_window", 9),
                "checkpoint_target": data_cfg.get("checkpoint_target", "best"),
                "checkpoint_path": data_cfg.get("checkpoint_path"),
            },
            "predict": {
                "batch_size": self.kwargs.get("batch_size", data_cfg.get("batch_size", 1)),
                "num_workers": self.kwargs.get("num_workers", data_cfg.get("num_workers", 0)),
                "split": self.kwargs.get("split", "test"),
                "threshold": self.kwargs.get("threshold", data_cfg.get("threshold")),
                "visualize": self.kwargs.get("visualize", data_cfg.get("visualize", False)),
                "source": self.kwargs.get("source", data_cfg.get("source")),
                "output_dir": self.kwargs.get("output_dir", data_cfg.get("output_dir")),
            },
            "export": {},
            "output": data_cfg.get("output", {}),
        }
        return predict_from_runtime_config(config)
