import numpy as np
import pytest
from unittest.mock import patch, mock_open
import pyRTX.utilities as utils
import datetime

def test_export_formatted():
    accelerations = np.array([[1, 2, 3], [4, 5, 6]])
    epochs = np.array([0, 1])
    filename = 'test.txt'

    with patch('numpy.savetxt') as mock_savetxt:
        utils.export_formatted(accelerations, epochs, filename, units='km/s^2')
        mock_savetxt.assert_called_once()

    with pytest.raises(ValueError):
        utils.export_formatted(np.array([1, 2, 3]), epochs, filename)

    with pytest.raises(ValueError):
        utils.export_formatted(accelerations, np.array([0]), filename)

def test_export_exac():
    satelliteID = 12345
    data = np.array([[1, 2, 3], [4, 5, 6]])
    tstep = 60
    startTime = datetime.datetime(2023, 1, 1, 0, 0, 0)
    endTime = datetime.datetime(2023, 1, 1, 0, 1, 0)
    outFileName = 'test.exac'

    with patch('scipy.io.FortranFile') as mock_fortranfile:
        utils.export_exac(satelliteID, data, tstep, startTime, endTime, outFileName)
        mock_fortranfile.assert_called_once_with(outFileName, 'w')

def test_to_datetime():
    epoch = 0.0
    with patch('spiceypy.et2utc', return_value='2000 JAN 01 12:00:00.000000'):
        dt = utils.to_datetime(epoch)
        assert isinstance(dt, datetime.datetime)

@patch('spiceypy.spkezr')
def test_getScPosVel(mock_spkezr):
    mock_spkezr.return_value = (np.array([[1, 2, 3, 4, 5, 6]]), 0)
    sc_pos, sc_vel = utils.getScPosVel('sc', 'center', 0, 'frame')
    assert sc_pos.shape == (1, 3)
    assert sc_vel.shape == (1, 3)
