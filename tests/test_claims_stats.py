import unittest

from hops.claims_stats import bootstrap_ci, paired_seed_bootstrap, spearman_trend


class TestClaimsStats(unittest.TestCase):
    def test_bootstrap_determinism(self) -> None:
        vals = [1.0, 2.0, 3.0]
        lo1, mid1, hi1 = bootstrap_ci(vals, rng_seed=123, n_boot=200, ci=0.9)
        lo2, mid2, hi2 = bootstrap_ci(vals, rng_seed=123, n_boot=200, ci=0.9)
        self.assertEqual((lo1, mid1, hi1), (lo2, mid2, hi2))

    def test_paired_seed_bootstrap(self) -> None:
        pairs = [(2.0, 1.0), (4.0, 2.0)]
        lo, mid, hi = paired_seed_bootstrap(pairs, rng_seed=0, n_boot=200, ci=0.9, effect="delta")
        self.assertTrue(lo <= mid <= hi)
        self.assertAlmostEqual(mid, 1.5, delta=0.5)

    def test_spearman_trend(self) -> None:
        x = [1.0, 2.0, 3.0]
        y = [1.0, 2.0, 3.0]
        rho, pvalue = spearman_trend(x, y)
        self.assertGreater(rho, 0.9)
        if pvalue is not None:
            self.assertLessEqual(pvalue, 0.1)


if __name__ == "__main__":
    unittest.main()
