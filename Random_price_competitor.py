from Competitor import Competitor
import numpy as np

class Random_price_competitor(Competitor):

	#store parameterdump here (until competition)

	def __init__(self, competitor_number):
		Competitor.__init__(self, competitor_number)
		np.random.seed(0)
	
	def p(self, prices_historical, demand_historical, t):#, parameterdump
	
		# if it's the first day
		#if demand_historical.size == 0:
		if t == 0:
			#store the number of competitors parameter in parameterdump
			self.C=prices_historical.size
		
		popt = np.random.uniform(0,100)
		return popt