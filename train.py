import os
import sys

ROOT = os.path.dirname(__file__)
src_path = os.path.join(ROOT, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from hops.sim import main


if __name__ == "__main__":
    main()
