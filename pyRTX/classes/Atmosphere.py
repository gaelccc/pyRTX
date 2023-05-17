import numpy as np
import os
import spiceypy as sp

class VenusGram():

	"""
	An utility class for providing a python interface to VenusGram2005
	
	Requirements:
	- python3
	- numpy
	- spiceypy
	- A compiled version of VenusGram2005	

	Note: This version has been tested only on UNIX machines

	The intended usage:
	
	vg = VenusGram()
	rho, T = vg.compute(epoch, lat, lon, height, inputUnits = 'deg')

	Parameters:
	*all these patameters can be either float or numpy  arrays of floats
	epoch: epoch output of spiceypy.str2et [The epoch must be provided in TDB. This can be changed modyfying 'epoch_converter' sub]
	lat: latitude in the units specified in inputUnits
	lon: longitude in the units specified in inputUnits
	height: height above the reference ellipsoid [km]

	Returns:
	rho: density in kg/m3
	T:   temperature in K

	"""
	

	def __init__(self, dataFolder = '/home/cascioli/VenusGram2005/', execFolder = '/home/cascioli/VenusGram2005/', tmpFolder = './tmp' ):
		"""
		Initialize the istance of the class
		
		Parameters:
		dataFolder: [str] the path to the data folder of VenusGram2005
		execFolder: [str] the path to where the venusgrm_V05.x executable is
		tmpFolder:  [str] the path to create a temporary convenience folder for Vgram input/output caching. Defaults to ./tmp

		"""
		
		if not os.path.exists(tmpFolder):
			os.system('mkdir {}'.format(tmpFolder))
		else:
			os.system('rm -r {}'.format(tmpFolder))
			os.system('mkdir {}'.format(tmpFolder))
		
		self.lstFile = tmpFolder + '/LIST.txt'
		self.outFile = tmpFolder + '/OUTPUT.txt'
		self.tmpFolder = tmpFolder
		self.execFolder = execFolder
		self.dataFolder = dataFolder









	def namelistWriter(self, epoch, lat, lon, height):
		f = open(self.tmpFolder + '/input.txt', 'w')

		M,D,Y,H,m,S = self.epoch_converter( epoch)

		txt =   f"""
$INPUT
  LSTFL     = '{self.tmpFolder}/LIST.txt'
  OUTFL     = '{self.tmpFolder}/OUTPUT.txt'
  TRAJFL    = 'TRAJDATA.txt'
  profile   = 'null'
  DATADIR   = '{self.dataFolder}'
  IERT      = 0
  IUTC      = 1
  Month     = {M}
  Mday      = {D}
  Myear     = {Y}
  Ihr       = {H}
  Imin      = {m}
  Sec       = {S}
  NPOS      = 1
  LonEast   = 1
  NR1       = 1234
  NVARX     = 1
  NVARY     = 0
  LOGSCALE  = 0
  FLAT      = {lat}
  FLON      = {lon}
  FHGT      = {height}
  DELHGT    = 1.0
  DELLAT    = 0.3
  DELLON    = 0.5
  DELTIME   = 500.0
  profnear  = 0.0
  proffar   = 0.0
  rpscale   = 1.0
  NMONTE    = 1
  iup       = 13
  corlmin   = 0.0
$END
"""

	

		f.write(txt)
		f.close()



	def epoch_converter(self, epoch, inputFormat = 'TDB', outputformat = 'UTC'):
		outPicture = '01 01 2011 01:20:30.000 {}'.format(outputformat)
		pic, ok, xerror = sp.tpictr(outPicture)
		if isinstance(epoch, float):
			outEp = sp.timout(epoch, pic)
	
		outEp = outEp.split(' ')
		M = int(outEp[0])
		D = int(outEp[1])
		Y = int(outEp[2])
		time = outEp[3].split(':')
		H = time[0]
		m = time[1]
		S = time[2]

		return M, D, Y, H, m, S


	def readResults(self):
		with open(self.tmpFolder + '/OUTPUT.txt', 'r') as f:
			dat = np.loadtxt(f, skiprows = 1)
			rho = dat[4]
			T = dat[5]
			CO2 = dat[11]
			N2 = dat[12]
			O = dat[13]
			CO = dat[14]
			He = dat[15]
			N = dat[16]
			H = dat[17]
		
		composition = [CO2, N2, O, CO, He, N, H]
		with open(os.getcwd()+'/TPresHgt.txt', 'r') as f:
			dat = np.loadtxt(f, skiprows = 1)
			P = dat[2]
		return rho, T, P, composition

	def readVariabilities(self, kind):
		'''
		Read results on variables with variabilities (e.g. Density)

		Input:
		kind : [str] The requested variable (available: Density)

		Output:

		low : [float] Variable low range
		avg : [float] Variable average range
		hig : [float] Variable high range

		'''

		with open(os.getcwd() + '/Density.txt', 'r') as f:
			dat = np.loadtxt(f, skiprows = 1)
			low = dat[1]
			avg = dat[2]
			hig = dat[3]
		return low, avg, hig

	def compute(self, epoch, lat, lon, height, inputUnits = 'deg', variabilities = None):


		if isinstance(epoch, float):
			epoch = [epoch]
		if isinstance(lat, float):
			lat = [lat]
		if isinstance(lon, float):
			lon = [lon]
		if isinstance(height, float):
			height = [height]



		#Check length of arrays
		if len(lat) != len(lon):
			raise ValueError('Latitude and Longitude arrays must be of same length')
		if len(lat) != len(epoch):
			raise ValueError('The epoch array must be of same lehgth of lat and lon')
		if len(height) != len(epoch):
			raise ValueError('The height array must be of same lehgth of lat and lon')

		if inputUnits == 'rad':
			conv = np.pi/180.0
			lat = lat * conv
			lon = lon * conv
	

		rho = np.zeros(len(epoch))
		T = np.zeros(len(epoch))
		P = np.zeros(len(epoch))
		composition = np.zeros((len(epoch), 7))
		if variabilities != None:
			var_low = np.zeros(len(epoch))
			var_avg = np.zeros(len(epoch))
			var_hig = np.zeros(len(epoch))

		for i, ep in enumerate(epoch):
			self.namelistWriter(ep, lat[i], lon[i], height[i])
			os.system( 'echo {} | {}venusgrm_V05.x 2&> {}/log.txt'.format(self.tmpFolder + '/input.txt', self.execFolder, self.tmpFolder))
			rho[i], T[i], P[i], composition[i,:] = self.readResults()
			if variabilities != None:
				var_low[i], var_avg[i], var_hig[i] = self.readVariabilities(variabilities)

				

		if variabilities == None:
			return rho, T, P,composition
		else:
			return rho, T, P, composition, var_low, var_avg, var_hig
			





#################################
# THIS IS AN EXAMPLE USAGE
################################
if __name__ == '__main__':

	import spiceypy as sp
	import matplotlib.pyplot as plt
	METAKR = 'spice/metakernel_veritas.tm'

	sp.furnsh(METAKR)


	vg = VenusGram()
	epoch = sp.str2et('01 JAN 2023 10:00:02.032 TDB')
	height = np.linspace(0, 300)
	epoch = [epoch]*len(height)
	lat = [0.0]*len(height)
	lon = [0.0]*len(height)
	
	#vg.namelistWriter(epoch, 3,4,100)
	rho, T, _,_ = vg.compute(epoch, lat, lon, height)	


	fig, ax = plt.subplots()

	ax.plot(rho, height, label = 'Density', color = 'b')
	ax2 = ax.twiny()
	ax2.plot(T, height, label = 'Temperature', color = 'r')
	sp.unload(METAKR)

	ax.set_xscale('log')

	ax.set_xlabel(r'Density [$kg/m^3$]', color = 'b')
	ax2.set_xlabel(r'Temperature [K]', color = 'r')
	ax.set_ylabel('Height above surface [km]')
	plt.show()
