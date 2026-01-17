import os
import runpy


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    script_path = os.path.join(root, "scripts", "sweep_param.py")
    if not os.path.exists(script_path):
        raise SystemExit("sweep_param.py not found; run from the repo root or use scripts/sweep_param.py")
    runpy.run_path(script_path, run_name="__main__")


if __name__ == "__main__":
    main()
