import math
import unittest

from hops.claims_data import canonicalize_rows, filter_implies_reached_tau


class TestClaimsData(unittest.TestCase):
    def test_canonicalization_and_reached_tau(self) -> None:
        rows = [
            {
                "final_S_eff": "10",
                "max_G_true": "0.2",
                "T_tau": "5",
                "censored": "false",
            }
        ]
        canon = canonicalize_rows(rows)[0]
        self.assertAlmostEqual(canon["final_welfare"], math.log1p(10.0))
        self.assertAlmostEqual(canon["max_gap_true_ratio"], 0.2)
        self.assertAlmostEqual(canon["time_to_tau"], 5.0)
        self.assertTrue(canon["reached_tau"])

    def test_filter_implies_reached_tau(self) -> None:
        self.assertTrue(filter_implies_reached_tau("reached_tau == true"))
        self.assertFalse(filter_implies_reached_tau("episode_idx >= 0"))


if __name__ == "__main__":
    unittest.main()
