from __future__ import annotations

from pathlib import Path

import yaml

from .parser import ForgeLabelParser
from .schema import ForgeRecord
from .validator import resolve_label_path, validate_dataset_config


class ForgeDataset:
    def __init__(self, yaml_path: str | Path, split: str = "train") -> None:
        self.yaml_path = Path(yaml_path).expanduser().resolve()
        with self.yaml_path.open("r", encoding="utf-8") as handle:
            self.cfg = yaml.safe_load(handle)
        if not isinstance(self.cfg, dict):
            raise TypeError(f"Expected dataset YAML at {self.yaml_path}")
        validate_dataset_config(self.cfg)
        self.root = Path(self.cfg["path"]).expanduser()
        if not self.root.is_absolute():
            self.root = (self.yaml_path.parent / self.root).resolve()
        self.task = self.cfg["task"]
        self.media = self.cfg["media"]
        self.split = split
        self.parser = ForgeLabelParser()
        self.records = self._load_split(split)

    def _load_split(self, split: str) -> list[ForgeRecord]:
        split_ref = self.cfg["splits"][split]
        split_path = (self.root / split_ref).resolve()
        records: list[ForgeRecord] = []
        for raw_line in split_path.read_text(encoding="utf-8").splitlines():
            rel_path = raw_line.strip()
            if not rel_path:
                continue
            media_path = (self.root / rel_path).resolve()
            label_path = resolve_label_path(self.root, self.cfg, rel_path)
            annotations = self.parser.parse(label_path, self.media, self.task)
            records.append(
                ForgeRecord(
                    media_path=str(media_path),
                    label_path=str(label_path),
                    media=self.media,
                    task=self.task,
                    annotations=annotations,
                )
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ForgeRecord:
        return self.records[index]
