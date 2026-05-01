from pathlib import Path

import torch

from vjepa_forge.engine.exporter import Exporter
from vjepa_forge.engine.model import ForgeModel


def test_anomaly_exporter_writes_onnx(monkeypatch, tmp_path: Path):
    written = {}

    def fake_export(model, sample, output_path, *, opset, dynamic_axes):
        written["shape"] = tuple(sample.shape)
        Path(output_path).write_text("fake", encoding="utf-8")
        return str(Path(output_path).resolve())

    monkeypatch.setattr("vjepa_forge.engine.exporter.export_to_onnx", fake_export)
    model = ForgeModel(
        {
            "task": "anomaly",
            "media": "video",
            "backbone": {"name": "vit_base", "use_sdpa": False, "modality_embedding": False},
            "image_size": 32,
            "num_frames": 4,
        },
        data={"task": "anomaly", "media": "video", "image_size": 32, "clip_len": 4},
    )
    result = Exporter().export(
        model,
        format="onnx",
        output_path=tmp_path / "anomaly.onnx",
        sample=torch.randn(1, 4, 3, 32, 32),
    )
    assert Path(result.output_path).exists()
    assert written["shape"] == (1, 4, 3, 32, 32)
