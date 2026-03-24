import unittest

from src.data.schema import UnifiedSample, build_metadata_frame, infer_severity_from_power_loss


class SchemaTests(unittest.TestCase):
    def test_infer_severity(self):
        self.assertEqual(infer_severity_from_power_loss(2.0), "low")
        self.assertEqual(infer_severity_from_power_loss(10.0), "medium")
        self.assertEqual(infer_severity_from_power_loss(18.0), "high")
        self.assertEqual(infer_severity_from_power_loss(25.0), "urgent")

    def test_build_metadata_frame(self):
        sample = UnifiedSample(
            sample_id="demo:1",
            dataset_name="deepsolareye",
            modality="rgb",
            image_path="image.jpg",
        )
        frame = build_metadata_frame([sample])
        self.assertEqual(len(frame), 1)
        self.assertIn("sample_id", frame.columns)
        self.assertIn("dataset_name", frame.columns)


if __name__ == "__main__":
    unittest.main()

