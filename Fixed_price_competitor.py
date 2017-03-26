from Competitor import Competitor

class Fixed_price_competitor(Competitor):

	#store parameterdump here (until competition)
	fixed_price=0

	def __init__(self, competitor_number, fixed_price):
		Competitor.__init__(self, competitor_number)
		self.fixed_price=fixed_price
	
	def p(self, prices_historical, demand_historical, t):#, parameterdump
	
		# if it's the first day
		#if demand_historical.size == 0:
		if t == 0:
			#store the number of competitors parameter in parameterdump
			self.C=prices_historical.size
		
		popt = self.fixed_price
		return popt