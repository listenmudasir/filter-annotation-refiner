import numpy as np
from refiner.quality import leakage_fraction, fragmentation_penalty


def test_leakage_and_fragmentation():
    m = np.zeros((100, 100), bool)
    m[20:60, 20:60] = True
    assert leakage_fraction(m, (15, 15, 65, 65), 100, 100) < 0.01
    m[80:85, 80:85] = True
    assert fragmentation_penalty(m, 4) > 0
