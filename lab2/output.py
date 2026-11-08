import csv
import json
from pathlib import Path
import statistics
import time


def write_json(path, data):
    def finite(obj):
        if isinstance(obj, float) and not float('-inf') < obj < float('inf'):
            return str(obj)
        if isinstance(obj, dict):
            return {str(k): finite(v) for k, v in obj.items()}
        if isinstance(obj, (tuple, list)):
            return [finite(v) for v in obj]
        return obj
    Path(path).write_text(json.dumps(finite(data), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def benchmark(fn, repeats, batch):
    fn()  # 预热，不保存到计时样本。
    samples = []
    for _ in range(repeats):
        begin = time.perf_counter_ns()
        for _ in range(batch):
            fn()
        samples.append((time.perf_counter_ns() - begin) / batch / 1e6)
    return dict(median_ms=statistics.median(samples), min_ms=min(samples), max_ms=max(samples)), samples
