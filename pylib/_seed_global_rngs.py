import random

from covasim import utils as cv_utils
import numpy as np


def seed_global_rngs(seed: int = 1) -> None:
    np.random.seed(seed)
    random.seed(seed)
    cv_utils.set_seed(seed)
