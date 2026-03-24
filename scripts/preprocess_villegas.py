from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.data.loaders import load_config, load_metadata_frame
from src.utils.io import ensure_dir
from src.utils.paths import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Villegas images into the processed directory.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    config = load_config(args.config)
    frame = load_metadata_frame(config["unified_metadata"]["csv"])
    subset = frame[frame["dataset_name"] == "villegas"].copy()
    out_dir = resolve_project_path(config["datasets"]["villegas"]["processed_dir"])
    ensure_dir(out_dir)

    for _, row in subset.iterrows():
        source = Path(row["image_path"])
        target = out_dir / source.name
        if not source.exists():
            continue
        image = Image.open(source).convert("RGB").resize((args.image_size, args.image_size))
        image.save(target)

    print(f"Preprocessed {len(subset)} Villegas images to {out_dir}")


if __name__ == "__main__":
    main()
