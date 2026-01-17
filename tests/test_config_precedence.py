import os
import sys
import types
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

HAS_DEPS = True
IMPORT_ERROR = ""
try:
    from hops.sim import EXPERIMENTS, apply_experiment_overrides, has_cfg
except ModuleNotFoundError as exc:
    if exc.name == "numpy":
        HAS_DEPS = False
        IMPORT_ERROR = "numpy not installed; activate venv to run config precedence tests"
    else:
        raise


@unittest.skipUnless(HAS_DEPS, IMPORT_ERROR)
class TestConfigPrecedence(unittest.TestCase):
    def test_preset_does_not_override_cli_flag(self):
        args = types.SimpleNamespace(exp_name="no_gov", gov_enabled=True)
        apply_experiment_overrides(args, explicit_flags={"gov_enabled"}, config_keys=set())
        self.assertTrue(args.gov_enabled)

    def test_preset_does_not_override_config_value(self):
        cfg = {"governance": {"enabled": False}}
        config_keys = set()
        if has_cfg(cfg, ("governance", "enabled")):
            config_keys.add("gov_enabled")
        args = types.SimpleNamespace(exp_name="baseline", gov_enabled=False)
        apply_experiment_overrides(args, explicit_flags=set(), config_keys=config_keys)
        self.assertFalse(args.gov_enabled)

    def test_preset_applies_when_unset(self):
        args = types.SimpleNamespace(exp_name="baseline", gov_enabled=None)
        apply_experiment_overrides(args, explicit_flags=set(), config_keys=set())
        self.assertTrue(args.gov_enabled)

    def test_preset_applies_for_missing_config_key(self):
        preset_name = "_test_preset_steps"
        EXPERIMENTS[preset_name] = {"episodes": 9, "steps_per_episode": 99}
        try:
            cfg = {"training": {"episodes": 3}}
            config_keys = set()
            if has_cfg(cfg, ("training", "episodes")):
                config_keys.add("episodes")
            self.assertFalse(has_cfg(cfg, ("training", "steps_per_episode")))
            if has_cfg(cfg, ("training", "steps_per_episode")):
                config_keys.add("steps")
            # Preset key steps_per_episode maps to CLI-facing args.steps.
            args = types.SimpleNamespace(exp_name=preset_name, episodes=3, steps=None)
            apply_experiment_overrides(args, explicit_flags=set(), config_keys=config_keys)
            self.assertEqual(args.episodes, 3)
            self.assertEqual(args.steps, 99)
        finally:
            EXPERIMENTS.pop(preset_name, None)


if __name__ == "__main__":
    unittest.main()
