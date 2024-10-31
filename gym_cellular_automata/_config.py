from pathlib import Path

import numpy as np

# Global path on current machine
PROJECT_PATH = Path(__file__).parents[1]


# Delegation of explicit typing as much as possible
# For floats using the spaces Box default
TYPE_BOX = np.float64
