import os
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if root not in sys.path:
    sys.path.insert(0, root)

from hops.package_manager import main


if __name__ == "__main__":
    main()
