import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

try:
    from src.data.datasets import DeepSolarEyeDataset
    from src.data.transforms_rgb import build_rgb_transform
except Exception:
    DeepSolarEyeDataset = None
    build_rgb_transform = None


@unittest.skipIf(DeepSolarEyeDataset is None or build_rgb_transform is None, "Torch or torchvision not available")
class DatasetTests(unittest.TestCase):
    def test_deepsolareye_dataset_item(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.jpg"
            Image.new("RGB", (64, 64), color=(128, 128, 128)).save(image_path)
            frame = pd.DataFrame(
                [
                    {
                        "sample_id": "deepsolareye:sample",
                        "dataset_name": "deepsolareye",
                        "modality": "rgb",
                        "image_path": str(image_path),
                        "power_loss_pct": 7.0,
                        "severity_label": "medium",
                        "irradiance": 800,
                        "temperature": 28,
                        "hour_sin": 0.0,
                        "hour_cos": 1.0,
                    }
                ]
            )
            dataset = DeepSolarEyeDataset(
                frame,
                transform=build_rgb_transform(64, is_train=False),
                tabular_features=["irradiance", "temperature", "hour_sin", "hour_cos"],
            )
            item = dataset[0]
            self.assertEqual(item["sample_id"], "deepsolareye:sample")
            self.assertEqual(tuple(item["image"].shape), (3, 64, 64))


if __name__ == "__main__":
    unittest.main()

