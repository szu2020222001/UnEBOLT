"""Evaluate a saved BOLD-to-PLV experiment without retraining."""
import sys
from engine_plv import main

if __name__ == "__main__":
    main([*sys.argv[1:], "--eval-only"])
