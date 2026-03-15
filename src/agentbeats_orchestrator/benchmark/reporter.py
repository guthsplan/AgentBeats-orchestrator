from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_benchmark_results(output_dir: str, results: dict[str, Any]) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_path = output_path / "benchmark_results.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    return str(result_path)
