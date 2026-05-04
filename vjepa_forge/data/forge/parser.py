from __future__ import annotations

from pathlib import Path

from .schema import ForgeAnnotation
from .validator import validate_task_media


class ForgeLabelParser:
    def parse(self, label_path: str | Path, media: str, task: str) -> list[ForgeAnnotation]:
        validate_task_media(task, media)
        path = Path(label_path)
        if not path.exists():
            return []
        annotations: list[ForgeAnnotation] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            annotations.append(self.parse_line(line, media=media, task=task))
        return annotations

    def parse_line(self, line: str, media: str, task: str) -> ForgeAnnotation:
        parts = line.split()
        op = parts[0]
        if op == "cls":
            return self._parse_cls(parts, media, task)
        if op == "det":
            return self._parse_det(parts, media, task)
        if op == "seg":
            return self._parse_seg(parts, media, task)
        if op in {"ano", "ano_box", "ano_seg"}:
            return self._parse_anomaly(parts, media, task)
        raise ValueError(f"Unknown label op: {op}")

    def _parse_cls(self, parts: list[str], media: str, task: str) -> ForgeAnnotation:
        if task != "classify":
            raise ValueError("cls labels are only valid for classify task")
        values = [int(value) for value in parts[1:]]
        if media == "image":
            if not values:
                raise ValueError("Image cls label requires at least one class id")
            return ForgeAnnotation("cls", {"class_ids": values})
        if len(values) == 1:
            return ForgeAnnotation("cls", {"class_id": values[0]})
        if len(values) == 3:
            return ForgeAnnotation("cls", {"class_id": values[0], "start_frame": values[1], "end_frame": values[2]})
        raise ValueError("Video cls label must be whole-video or ranged")

    def _parse_det(self, parts: list[str], media: str, task: str) -> ForgeAnnotation:
        if task != "detect":
            raise ValueError("det labels are only valid for detect task")
        if media == "image":
            if len(parts) != 6:
                raise ValueError("Image det label must have 6 fields")
            return ForgeAnnotation(
                "det",
                {
                    "class_id": int(parts[1]),
                    "box": [float(value) for value in parts[2:6]],
                },
            )
        if len(parts) not in {7, 8}:
            raise ValueError("Video det label must have 7 or 8 fields")
        payload = {
            "frame_idx": int(parts[1]),
            "class_id": int(parts[2]),
            "box": [float(value) for value in parts[3:7]],
        }
        if len(parts) == 8:
            payload["track_id"] = int(parts[7])
        return ForgeAnnotation("det", payload)

    def _parse_seg(self, parts: list[str], media: str, task: str) -> ForgeAnnotation:
        if task != "segment":
            raise ValueError("seg labels are only valid for segment task")
        if media == "image":
            if len(parts) < 4 or len(parts[2:]) % 2 != 0:
                raise ValueError("Image seg label must include polygon pairs")
            return ForgeAnnotation(
                "seg",
                {
                    "class_id": int(parts[1]),
                    "polygon": [float(value) for value in parts[2:]],
                },
            )
        if len(parts) < 5:
            raise ValueError("Video seg label must include frame and class")
        payload = {
            "frame_idx": int(parts[1]),
            "class_id": int(parts[2]),
        }
        start_idx = 3
        if len(parts[3:]) % 2 != 0:
            payload["object_id"] = int(parts[3])
            start_idx = 4
        polygon = [float(value) for value in parts[start_idx:]]
        if len(polygon) < 4 or len(polygon) % 2 != 0:
            raise ValueError("Video seg polygon must include coordinate pairs")
        payload["polygon"] = polygon
        return ForgeAnnotation("seg", payload)

    def _parse_anomaly(self, parts: list[str], media: str, task: str) -> ForgeAnnotation:
        if task != "anomaly":
            raise ValueError("anomaly labels are only valid for anomaly task")
        op = parts[0]
        if op == "ano":
            if parts[1] == "normal":
                return ForgeAnnotation("ano", {"status": "normal"})
            if media == "image":
                if len(parts) != 3:
                    raise ValueError("Image abnormal label must include class_id")
                return ForgeAnnotation("ano", {"status": "abnormal", "class_id": int(parts[2])})
            if len(parts) != 5:
                raise ValueError("Video abnormal label must include frame range and class_id")
            return ForgeAnnotation(
                "ano",
                {
                    "status": "abnormal",
                    "start_frame": int(parts[2]),
                    "end_frame": int(parts[3]),
                    "class_id": int(parts[4]),
                },
            )
        if op == "ano_box":
            if media == "image":
                return ForgeAnnotation("ano_box", {"class_id": int(parts[1]), "box": [float(value) for value in parts[2:6]]})
            return ForgeAnnotation(
                "ano_box",
                {
                    "frame_idx": int(parts[1]),
                    "class_id": int(parts[2]),
                    "box": [float(value) for value in parts[3:7]],
                },
            )
        if op == "ano_seg":
            if media == "image":
                return ForgeAnnotation("ano_seg", {"class_id": int(parts[1]), "polygon": [float(value) for value in parts[2:]]})
            return ForgeAnnotation(
                "ano_seg",
                {
                    "frame_idx": int(parts[1]),
                    "class_id": int(parts[2]),
                    "polygon": [float(value) for value in parts[3:]],
                },
            )
        raise ValueError(f"Unknown anomaly op: {op}")
