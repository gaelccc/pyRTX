from unittest.mock import patch
import pyRTX.helpers as helpers

@patch('subprocess.run')
def test_get_download_agent(mock_run):
    # Test case 1: wget is installed
    mock_run.return_value.stdout = 'GNU Wget 1.20.3'
    assert helpers.get_download_agent() == 'wget'

    # Test case 2: wget is not installed
    mock_run.side_effect = FileNotFoundError
    assert helpers.get_download_agent() == 'curl'

    # Test case 3: wget is installed but not GNU wget
    mock_run.side_effect = None
    mock_run.return_value.stdout = 'Some other wget'
    assert helpers.get_download_agent() == 'curl'
