import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

HAS_MPL = True
try:
    import matplotlib  # noqa: F401
except Exception:
    HAS_MPL = False


def _write_package_yaml(path: str, package_name: str, plot_id: str) -> None:
    contents = f"""package:
  name: "{package_name}"
  app_version: "0.1.0"
  config: "config.yaml"
  analysis:
    strict: true
    output_dir: "analysis/package_compare"
    plots:
      - {plot_id}
    tables:
      - combined_episode_points_csv
      - package_summary_csv
      - package_summary_json
    always_include:
      - analysis_manifest_json

jobs:
  - name: "smoke_suite"
    kind: "suite"
    mode: "simulate"
    exp_names: [baseline, no_gov]
    seeds: [0]
    episodes: 1
    steps: 5
    log_mode: "metrics_only"
    log_stride: 1
    analyze: true
    zip: false
    analysis:
      plots: []
      strict: false

  - name: "smoke_sweep"
    kind: "sweep"
    mode: "simulate"
    exp_name: "baseline"
    param_name: "sneakiness.uniform_max"
    param_min: 0.0
    param_max: 0.1
    param_n: 2
    param_scale: "linear"
    seeds: [0]
    episodes: 1
    steps: 5
    log_mode: "metrics_only"
    log_stride: 1
    analyze: true
    zip: false
    analysis:
      plots: []
      strict: false
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(contents)


class TestPackageSmoke(unittest.TestCase):
    def _run_package(self, yaml_path: str) -> subprocess.CompletedProcess:
        cmd = ["hops-package", "--package", yaml_path, "--allow_dirty"]
        if shutil.which("hops-package") is None:
            cmd = [
                sys.executable,
                os.path.join("scripts", "run_package.py"),
                "--package",
                yaml_path,
                "--allow_dirty",
            ]
        return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)

    def test_smoke_package_manifest(self):
        package_name = f"smoke_pkg_{uuid.uuid4().hex[:8]}"
        if not HAS_MPL:
            self.skipTest("matplotlib not installed; package analyze step requires it")
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = os.path.join(tmpdir, "smoke_pkg.yaml")
            _write_package_yaml(yaml_path, package_name, "package_pareto_frontier")
            result = self._run_package(yaml_path)
            self.assertEqual(
                result.returncode,
                0,
                msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
        manifest_path = os.path.join(
            REPO_ROOT,
            "runs",
            "packages",
            package_name,
            "analysis",
            "analysis_manifest.json",
        )
        self.assertTrue(os.path.exists(manifest_path))
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        produced_tables = manifest.get("produced", {}).get("tables", [])
        statuses = {t.get("status") for t in produced_tables}
        self.assertTrue({"ok", "failed", "skipped"} & statuses)

    def test_strict_unknown_plot_writes_manifest(self):
        package_name = f"smoke_pkg_unknown_{uuid.uuid4().hex[:8]}"
        if not HAS_MPL:
            self.skipTest("matplotlib not installed; package analyze step requires it")
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = os.path.join(tmpdir, "smoke_pkg_unknown.yaml")
            _write_package_yaml(yaml_path, package_name, "unknown_plot_id")
            result = self._run_package(yaml_path)
            self.assertEqual(
                result.returncode,
                2,
                msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
        manifest_path = os.path.join(
            REPO_ROOT,
            "runs",
            "packages",
            package_name,
            "analysis",
            "analysis_manifest.json",
        )
        self.assertTrue(os.path.exists(manifest_path))
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest.get("exit_status"), 1)
        errors = manifest.get("errors", [])
        self.assertTrue(
            any(
                err.get("reason") == "unknown_plot_id" and err.get("plot_id") == "unknown_plot_id"
                for err in errors
            )
        )

    def test_list_plots_tables(self):
        cmd = ["hops-package", "--list_plots"]
        if shutil.which("hops-package") is None:
            cmd = [sys.executable, os.path.join("scripts", "run_package.py"), "--list_plots"]
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertTrue(result.stdout.strip())

        cmd = ["hops-package", "--list_tables"]
        if shutil.which("hops-package") is None:
            cmd = [sys.executable, os.path.join("scripts", "run_package.py"), "--list_tables"]
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertTrue(result.stdout.strip())

    def test_readme_smoke_command(self):
        cmd = [
            "hops-package",
            "--package",
            os.path.join("configs", "examples", "package_smoke.yaml"),
            "--allow_dirty",
            "--dry_run",
        ]
        if shutil.which("hops-package") is None:
            cmd = [
                sys.executable,
                os.path.join("scripts", "run_package.py"),
                "--package",
                os.path.join("configs", "examples", "package_smoke.yaml"),
                "--allow_dirty",
                "--dry_run",
            ]
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
