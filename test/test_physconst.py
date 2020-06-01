"""
Tests for ebisim.physconst
"""
# pylint: disable=missing-function-docstring

import pytest

import ebisim.physconst

def test_elementary_charge():
    assert ebisim.physconst.Q_E == pytest.approx(1.60217662e-19)

def test_electron_mass():
    assert ebisim.physconst.M_E == pytest.approx(9.10938356e-31)

def test_pi():
    assert ebisim.physconst.PI == pytest.approx(3.14159265359)

def test_vacuum_permittivity():
    assert ebisim.physconst.EPS_0 == pytest.approx(8.854187817e-12)

def test_boltzmann_constant():
    assert ebisim.physconst.K_B == pytest.approx(1.38064852e-23)

def test_speed_of_light():
    assert ebisim.physconst.C_L == 299792458

def test_fine_structure_constant():
    assert ebisim.physconst.ALPHA == pytest.approx(0.007297352566)

def test_reduced_planck_constant():
    assert ebisim.physconst.HBAR == pytest.approx(1.054571800e-34)

def test_proton_mass():
    assert ebisim.physconst.M_P == pytest.approx(1.6726219e-27)

def test_electron_mass_electronvolts():
    assert ebisim.physconst.M_E_EV == pytest.approx(510998.9461)

def test_rydberg_energy_electronvolts():
    assert ebisim.physconst.RY_EV == pytest.approx(13.605693009)

def test_reduced_electron_compton_wavelength():
    assert ebisim.physconst.COMPT_E_RED == pytest.approx(2.4263102367e-12 / 3.14159265359)
