import sys

from . import analyze, run_experiment, run_suite, sim, sweep_param


def main() -> None:
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "analyze":
            sys.argv.pop(1)
            analyze.main()
            return
        if cmd == "simulate":
            sys.argv.pop(1)
            if "--mode" not in sys.argv:
                sys.argv.extend(["--mode", "simulate"])
            sim.main()
            return
        if cmd == "run":
            sys.argv.pop(1)
            run_experiment.main()
            return
        if cmd in {"suite", "run_suite"}:
            sys.argv.pop(1)
            run_suite.main()
            return
        if cmd in {"sweep", "sweep_param"}:
            sys.argv.pop(1)
            sweep_param.main()
            return
        if cmd == "train":
            sys.argv.pop(1)
            if "--mode" not in sys.argv:
                sys.argv.extend(["--mode", "train"])
            sim.main()
            return
    sim.main()


if __name__ == "__main__":
    main()
