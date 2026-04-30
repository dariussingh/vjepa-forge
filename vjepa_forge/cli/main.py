from __future__ import annotations

import sys

from . import benchmark, export, predict, train, val


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: vjepa-forge <train|val|predict|export|benchmark> recipe=...")
    command = sys.argv[1]
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    if command == "train":
        train.main()
        return
    if command == "val":
        val.main()
        return
    if command == "predict":
        predict.main()
        return
    if command == "export":
        export.main()
        return
    if command == "benchmark":
        benchmark.main()
        return
    raise SystemExit(f"Unknown command: {command}")
