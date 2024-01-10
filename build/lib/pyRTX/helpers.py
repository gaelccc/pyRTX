# Helper functions
import subprocess

def get_download_agent():
	"""
	Is wget installed? This helper function returns the download program available in the system.
	This function is used for automated download of 'example data'
	Parameters:
	

	Returns:
	agent: [str] 'wget' if wget is installed, 'curl' otherwise

	To Do:
	Currently this function works only with UNIX/macOS systems. Expand it to windows systems
	"""


	try:
		result = subprocess.run(['wget', '--version'], capture_output=True, text=True)
		output = result.stdout

		if 'GNU Wget' in output:
			return 'wget'
		else:
			return 'curl'
	except FileNotFoundError:
		return 'curl'





