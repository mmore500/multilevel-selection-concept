import random

import numpy as np


def seed_global_rngs(seed: int = 1) -> None:
    np.random.seed(seed)
    random.seed(seed)
