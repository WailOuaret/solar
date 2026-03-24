from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.inference.pipeline import InferencePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end inference for one image.")
    parser.add_argument("--image", required=True, help="Image path.")
    parser.add_argument("--modality", required=True, choices=["rgb", "thermal"])
    parser.add_argument("--config", default="configs/inference.yaml")
    args = parser.parse_args()

    pipeline = InferencePipeline(config_path=args.config)
    result = pipeline.predict_sample(args.image, modality=args.modality)
    print(pd.Series(result).to_string())


if __name__ == "__main__":
    main()
