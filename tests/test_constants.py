import pyRTX.constants as const

def test_c():
    assert const.c == 299792.458

def test_stefan_boltzmann():
    assert const.stefan_boltzmann == 5.670367e-8

def test_au():
    assert const.au == 149597870.700

def test_unit_conversions():
    assert const.unit_conversions['m'] == 1e-3
    assert const.unit_conversions['km'] == 1.0
