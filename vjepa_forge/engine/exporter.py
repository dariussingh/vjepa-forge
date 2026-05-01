from __future__ import annotations


class Exporter:
    def export(self, model, *, format: str = "onnx") -> dict[str, str]:
        return {"format": format, "status": "not_implemented"}
