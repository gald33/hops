import os
import runpy


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    script_path = os.path.join(root, "scripts", "run_suite.py")
    if not os.path.exists(script_path):
        raise SystemExit("run_suite.py not found; run from the repo root or use scripts/run_suite.py")
    runpy.run_path(script_path, run_name="__main__")


if __name__ == "__main__":
    main()
