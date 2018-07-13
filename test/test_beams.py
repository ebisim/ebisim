"""
Tests for ebisim.beams
"""
#TODO: extend test suite

import ebisim.beams
import pytest

def test_electron_velocity():
    assert ebisim.beams.electron_velocity(3e3) == pytest.approx(3.234302e7)
    assert ebisim.beams.electron_velocity(3e4) == pytest.approx(9.84447e7)
    assert ebisim.beams.electron_velocity(3e5) == pytest.approx(2.327965e8)
