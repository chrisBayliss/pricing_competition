from Competitor import Competitor
import numpy as np
import math as Math

class Sine_competitor(Competitor):

	#store parameterdump here (until competition)
	#A+B*sin(C*t)
	A=50.0
	B=50.0#0#
	D=0.02
	

	def __init__(self, competitor_number):
		Competitor.__init__(self, competitor_number)
		np.random.seed(0)
	
	def p(self, prices_historical, demand_historical, t):#, parameterdump
	
		if t == 0:
			#store the number of competitors parameter in parameterdump
			self.C=len(prices_historical)
			
			
			popt = self.A+self.B*Math.sin(self.D*t)
		else:
			popt = self.A+self.B*Math.sin(self.D*t)
			
		
		return popt