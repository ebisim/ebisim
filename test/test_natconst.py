import ebisim.natconst
import pytest

def test_elementary_charge():
    assert ebisim.natconst.Q_E == pytest.approx(1.60217662e-19)

def test_electron_mass():
    assert ebisim.natconst.M_E == pytest.approx(9.10938356e-31)

def test_pi():
    assert ebisim.natconst.PI == pytest.approx(3.14159265359)

def test_vacuum_permittivity():
    assert ebisim.natconst.EPS_0 == pytest.approx(8.854187817e-12)

def test_boltzmann_constant():
    assert ebisim.natconst.K_B == pytest.approx(1.38064852e-23)

def test_speed_of_light():
    assert ebisim.natconst.C_L == 299792458

def test_fine_structure_constant():
    assert ebisim.natconst.ALPHA == pytest.approx(0.007297352566)

def test_reduced_planck_constant():
    assert ebisim.natconst.HBAR == pytest.approx(1.054571800e-34)

def test_electron_mass_electronvolts():
    assert ebisim.natconst.M_E_EV == pytest.approx(510998.9461)

def test_rydberg_energy_electronvolts():
    assert ebisim.natconst.RY_EV == pytest.approx(13.605693009)

def test_reduced_electron_compton_wavelength():
    assert ebisim.natconst.COMPT_E_RED == pytest.approx(2.4263102367e-12 / 3.14159265359)
