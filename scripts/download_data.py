from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.utils.io import dump_yaml, ensure_dir
from src.utils.paths import resolve_project_path
from src.data.loaders import load_config


DATASET_SOURCES = {
    "deepsolareye": {
        "name": "DeepSolarEye",
        "url": "https://deep-solar-eye.github.io/",
        "license": "Creative Commons release referenced by dataset page",
        "notes": "Download may require following the project links from the official page.",
    },
    "villegas": {
        "name": "PV Panel Partial Shading & Electrical Dataset",
        "url": "https://www.mdpi.com/2306-5729/7/6/82",
        "license": "CC BY 4.0",
        "notes": "The paper references the dataset DOI. Raw files may require manual retrieval.",
    },
    "trsai": {
        "name": "Thermal Hotspot Detection Dataset",
        "url": "https://data.mendeley.com/datasets/8sxfmrpfpv/1",
        "license": "CC BY 4.0",
        "notes": "Mendeley datasets can require manual download from the dataset page.",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(dataset_name: str, raw_dir: Path) -> dict:
    files = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        files.append(
            {
                "path": str(path.relative_to(raw_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "dataset_name": dataset_name,
        "raw_dir": str(raw_dir),
        "file_count": len(files),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset directories and manifest files.")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data config.")
    args = parser.parse_args()

    config = load_config(args.config)
    docs_root = resolve_project_path("docs/dataset_cards")

    for dataset_key, dataset_cfg in config["datasets"].items():
        raw_dir = resolve_project_path(dataset_cfg["raw_dir"])
        ensure_dir(raw_dir)

        source = DATASET_SOURCES[dataset_key]
        dataset_card = {
            "dataset": source["name"],
            "dataset_key": dataset_key,
            "source_url": source["url"],
            "license": source["license"],
            "raw_dir": str(raw_dir),
            "notes": source["notes"],
            "status": "awaiting_download" if not any(raw_dir.iterdir()) else "files_present",
        }

        dump_yaml(dataset_card, docs_root / f"{dataset_key}_dataset_card.yaml")

        manifest_path = docs_root / f"{dataset_key}_manifest.yaml"
        if any(raw_dir.iterdir()):
            dump_yaml(build_manifest(dataset_key, raw_dir), manifest_path)
        else:
            dump_yaml({"dataset_name": dataset_key, "raw_dir": str(raw_dir), "file_count": 0, "files": []}, manifest_path)

    print("Dataset directories and manifests prepared.")
    print("If files are still missing, download the datasets manually into data/raw/<dataset_name>/ and rerun this script.")


if __name__ == "__main__":
    main()
