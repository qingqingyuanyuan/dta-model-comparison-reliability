#!/usr/bin/env python3
"""Download DeepDTA raw reference files.

These raw files are retained for provenance/reference only. The finalized
ZhiYao-Graph experiments currently use the BatchDTA processed source CSV files
under ``data/colab/BatchDTA_processed_data`` and then run
``scripts/prepare_final_datasets.py``.

Running this downloader alone does NOT create final training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data" / "raw"

URLS = {
    "kiba_drugs": (
        "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/KIBA/drugs.txt",
        "kiba",
    ),
    "kiba_targets": (
        "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/KIBA/targets.txt",
        "kiba",
    ),
    "kiba_Y": (
        "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/KIBA/Y.txt",
        "kiba",
    ),
    "davis_drugs": (
        "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis/drugs.txt",
        "davis",
    ),
    "davis_targets": (
        "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis/targets.txt",
        "davis",
    ),
    "davis_Y": (
        "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis/Y.txt",
        "davis",
    ),
}


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, save_path: Path):
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = save_path.with_suffix(save_path.suffix + ".part")

    try:
        with tmp.open("wb") as f, tqdm(
            desc=save_path.name,
            total=total,
            unit="B",
            unit_scale=True,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))
        tmp.replace(save_path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def download_datasets(
    datasets=("kiba", "davis"),
    *,
    overwrite=False,
    base_dir=BASE_DIR,
):
    base_dir = Path(base_dir)
    wanted = {str(x).lower() for x in datasets}
    unknown = wanted - {"kiba", "davis"}
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")

    manifest = {
        "role": "raw_reference_only",
        "note": (
            "Final experiments do not train directly from these files. "
            "Use BatchDTA processed CSVs + prepare_final_datasets.py."
        ),
        "files": {},
    }

    for name, (url, dataset) in URLS.items():
        if dataset not in wanted:
            continue

        save_path = base_dir / dataset / Path(url).name
        if save_path.exists() and not overwrite:
            print(f"[跳过] {save_path}")
        else:
            print(f"[下载] {name}")
            download_file(url, save_path)

        manifest["files"][str(save_path.relative_to(base_dir))] = {
            "url": url,
            "bytes": save_path.stat().st_size,
            "sha256": sha256(save_path),
        }

    manifest_path = base_dir / "raw_reference_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")
    print(
        "\n提醒：这些是原始参考文件，不是当前最终训练入口。"
        "\n正式数据请运行 scripts/prepare_final_datasets.py。"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        nargs="+",
        choices=["kiba", "davis"],
        default=["kiba", "davis"],
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--base-dir", type=Path, default=BASE_DIR)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_datasets(
        args.dataset,
        overwrite=args.overwrite,
        base_dir=args.base_dir,
    )
