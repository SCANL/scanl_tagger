import os
import sys

# Tests import `src` and `version` from the repository root, the same way `main` does.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
