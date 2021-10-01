import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
	long_description = fh.read()

setuptools.setup(
	name = 'pyRTX',
	version = '0.0.1',
	author = 'Gael Cascioli',
	author_email = 'gael.cascioli@uniroma1.it',
	description = 'A collection of tools for non-gravitational acceleration computation leveraging ray tracing techniques',
	url = '',
	packages = ['pyRTX'],
	classifiers = [
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
	python_requires='~=3.8',
	install_requires = [ 'numpy'],
	)

