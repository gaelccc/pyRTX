import numpy as np
from pyRTX.analysis_utils import LookupTableND
import spiceypy as sp

class Drag():
	def __init__(self, spacecraft, crossectionLUT, density, CD, body):

		"""
		Initialize the Drag instance

		Parameters:
		Spacecraft: Spacecraft object
		crossectionLUT: str or LookupTableND
		density: function that takes as input the height and returns the density in kg/km**2
		CD: [float] the object CD

		"""
		

		self.scName = spacecraft.name
		self.scFrame = spacecraft.base_frame
		if isinstance(crossectionLUT, LookupTableND):
			self.LUT = crossectionLUT
		elif isinstance(crossectionLUT, str):
			lut = np.load(LUT)
			y = np.linspace(-90,90, lut.shape[1])*np.pi/180
			x = np.linspace(0, 360, lut.shape[0])*np.pi/180
			self.LUT = LookupTableND(axes = (x,y), values = lut)
		else:
			raise TypeError('Error: the input lookup table must be either analysis_utols.LookupTableND or a file path to the numpy NDarray')


		if not callable(density):
			raise TypeError('Error: the input density must be a callable function')


		self.density = density
		self.body = body
		self.CD = CD

	
	def compute(self,epoch, frame = ''):
		"""
		Compute the drag at epoch in the spacecraft body frame
		"""

		st = sp.spkezr(self.scName, epoch, 'IAU_%s' %self.body.upper(), 'LT+S', self.body)
		rot = sp.pxform('IAU_%s' %self.body.upper(), self.scFrame, epoch)
		st_r = rot@st[0][0:3]
		vel_r = rot@st[0][3:6]

		_, dir1, dir2 = sp.recrad(-vel_r)
		h = np.linalg.norm(st[0][0:3])
		unitv = vel_r/np.linalg.norm(vel_r)
		rho = self.density(h)

		drag = - 0.5*rho*self.CD*np.linalg.norm(vel_r)**2*unitv*self.LUT[dir1, dir2]


		if frame != '' and frame.lower() != self.scFrame:
			rot = sp.pxform(self.scFrame, frame, epoch)
			drag = rot*drag
		return drag, vel_r
