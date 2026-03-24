import unittest

from src.inference.decision_rules import build_final_risk_score, recommend_action


class PipelineLogicTests(unittest.TestCase):
    def test_build_final_risk_score(self):
        score = build_final_risk_score(power_loss_pct=15.0, electrical_score=0.5, hotspot_probability=0.2)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_recommend_action_for_hotspot(self):
        result = recommend_action(power_loss_pct=4.0, electrical_score=0.1, hotspot_probability=0.9)
        self.assertEqual(result["final_severity"], "urgent")


if __name__ == "__main__":
    unittest.main()

