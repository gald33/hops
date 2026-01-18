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


def _run_package(yaml_path: str) -> subprocess.CompletedProcess:
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


def _write_fail_package_yaml(path: str, package_name: str) -> None:
    contents = f"""package:
  name: \"{package_name}\"
  app_version: \"0.1.2\"
  config: \"config.yaml\"
  analysis:
    strict: true
    output_dir: \"analysis/package_compare\"
    plots:
      - package_pareto_frontier
    tables:
      - combined_episode_points_csv
      - package_summary_csv
      - package_summary_json
    always_include:
      - analysis_manifest_json
  claims:
    strict: true
    output_dir: \"analysis/claims\"
    claims:
      - id: C_FAIL
        scope: package
        description: \"Forced failure for test\"
        target:
          kind: single
        metrics:
          - name: max_gap_true_ratio
            agg: max
            op: \"<=\"
            value: -1.0

jobs:
  - name: \"smoke_suite\"
    kind: \"suite\"
    mode: \"simulate\"
    exp_names: [baseline, no_gov]
    seeds: [0]
    episodes: 1
    steps: 5
    log_mode: \"metrics_only\"
    log_stride: 1
    analyze: true
    zip: false
    analysis:
      plots: []
      strict: false

  - name: \"smoke_sweep\"
    kind: \"sweep\"
    mode: \"simulate\"
    exp_name: \"baseline\"
    param_name: \"sneakiness.uniform_max\"
    param_min: 0.0
    param_max: 0.1
    param_n: 2
    param_scale: \"linear\"
    seeds: [0]
    episodes: 1
    steps: 5
    log_mode: \"metrics_only\"
    log_stride: 1
    analyze: true
    zip: false
    analysis:
      plots: []
      strict: false
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(contents)


class TestClaimsIntegration(unittest.TestCase):
    def test_claims_manifest_written(self) -> None:
        if not HAS_MPL:
            self.skipTest("matplotlib not installed; package analyze step requires it")
        yaml_path = os.path.join(REPO_ROOT, "configs", "examples", "package_smoke_claims.yaml")
        result = _run_package(yaml_path)
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        manifest_path = os.path.join(
            REPO_ROOT,
            "runs",
            "packages",
            "smoke_pkg_claims",
            "analysis",
            "claims",
            "claims_manifest.json",
        )
        self.assertTrue(os.path.exists(manifest_path))
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        statuses = {res.get("status") for res in manifest.get("results", [])}
        self.assertTrue({"supported", "inconclusive", "skipped", "rejected"} & statuses)

    def test_strict_claims_failure_writes_manifest(self) -> None:
        if not HAS_MPL:
            self.skipTest("matplotlib not installed; package analyze step requires it")
        package_name = f"smoke_claims_fail_{uuid.uuid4().hex[:8]}"
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = os.path.join(tmpdir, "claims_fail.yaml")
            _write_fail_package_yaml(yaml_path, package_name)
            result = _run_package(yaml_path)
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
            "claims",
            "claims_manifest.json",
        )
        self.assertTrue(os.path.exists(manifest_path))
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest.get("exit_status"), 2)
        self.assertTrue(any(r.get("claim_id") == "C_FAIL" for r in manifest.get("results", [])))


if __name__ == "__main__":
    unittest.main()
