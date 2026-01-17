import os
import tempfile
import unittest

from hops.package_analyze import build_analysis_manifest


class TestAnalysisContract(unittest.TestCase):
    def _base_kwargs(self, tmpdir):
        return {
            "package_dir": tmpdir,
            "package_name": "pkg",
            "app_version": "0.1.0",
            "config_hash": "hash",
            "git_commit": "deadbeef",
            "is_dirty": False,
            "package_outputs": {"produced": {}},
        }

    def test_unknown_plot_strict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = build_analysis_manifest(
                analysis_config={"plots": ["unknown_plot"], "strict": True},
                job_results=[],
                **self._base_kwargs(tmpdir),
            )
            self.assertEqual(manifest.get("exit_status"), 1)
            self.assertTrue(manifest.get("errors"))
            manifest_path = os.path.join(tmpdir, "analysis", "analysis_manifest.json")
            self.assertTrue(os.path.exists(manifest_path))

    def test_unknown_plot_non_strict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = build_analysis_manifest(
                analysis_config={"plots": ["unknown_plot"], "strict": False},
                job_results=[],
                **self._base_kwargs(tmpdir),
            )
            self.assertEqual(manifest.get("exit_status"), 0)
            self.assertFalse(manifest.get("errors"))
            self.assertTrue(manifest.get("missing", {}).get("package"))

    def test_not_applicable_plot_strict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job_results = [
                {
                    "job_name": "job",
                    "job_kind": "sweep",
                    "job_root": tmpdir,
                    "analysis": {"plots": ["gap_true_vs_observed_t"], "strict": True},
                }
            ]
            manifest = build_analysis_manifest(
                analysis_config={"plots": [], "strict": False},
                job_results=job_results,
                **self._base_kwargs(tmpdir),
            )
            self.assertEqual(manifest.get("exit_status"), 1)
            self.assertTrue(manifest.get("errors"))


if __name__ == "__main__":
    unittest.main()
