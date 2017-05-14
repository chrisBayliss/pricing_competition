from Competitor import Competitor
import numpy as np
import math as Math

class Mode_price_forecast_competitor(Competitor):

	#store parameterdump here (until competition)
	Base_value=[]
	Trend=[]
	prices_next_t=[]
	alpha=0.2
	beta=0.2
	
	mode_intervals=[10,20,30,40,50,60,70,80,90,100]
	mode_interval_size=10
	mode_interval_frequencies=[0]*10
	
	other_prices=[]

	def __init__(self, competitor_number):
		Competitor.__init__(self, competitor_number)
		np.random.seed(0)
	
	def p(self, prices_historical, demand_historical, t):#, parameterdump
	
		if t == 0:
			#store the number of competitors parameter in parameterdump
			self.C=len(prices_historical)
			
			self.Base_value=[[0 for j in range(self.C-1)] for i in range(self.T)]
			self.Trend=[[0 for j in range(self.C-1)] for i in range(self.T)]
			
			self.prices_next_t=[0 for i in range(self.C-1)]
			
			self.other_prices=[0]*(self.C-1)
			
			popt = np.random.uniform(0,100)
		elif t<2:
			#initialise base value (trend initialised to 0)
			#for c in range(self.C):
			counter=0
			for i in range(self.C):
				if i!=self.competitor_number:
					self.other_prices[counter]=prices_historical[i,t-1]
					counter=counter+1
			
			[sorted_prices, ind_order]=self.sort(self.other_prices)#prices_historical[:,t-1]
			self.Base_value[0]=sorted_prices
			
			#random initial price
			popt = np.random.uniform(0,100)
		else:
			
			#prices offered in the previous time period
			counter=0
			for i in range(self.C):
				if i!=self.competitor_number:
					self.other_prices[counter]=prices_historical[i,t-1]
					counter=counter+1
			
			[sorted_prices, ind_order]=self.sort(self.other_prices)#prices_historical[:,t-1]
			
			
			forecast_price_set=self.update_exp_smooth_params_return_forecast_prices(sorted_prices, t)
			
			#mode interval frequencies
			#reset
			for i in range(len(self.mode_interval_frequencies)):
				self.mode_interval_frequencies[i]=0
				
			for i in range(len(forecast_price_set)):
				interval=min(len(self.mode_interval_frequencies)-1, max(0, Math.floor(forecast_price_set[i]/self.mode_interval_size)))
				self.mode_interval_frequencies[interval]=self.mode_interval_frequencies[interval]+1
			
			#find the mode price
			max_frequency=0
			mode_price_index=None
			for i in range(len(self.mode_interval_frequencies)):
				if self.mode_interval_frequencies[i]>max_frequency:
					max_frequency=self.mode_interval_frequencies[i]
					mode_price_index=i
			
			popt = self.mode_intervals[mode_price_index]
		
		return popt
		
	#sorted prices could provide a more stable model, especially as competitor prices will not in general be modelled well with an exponential smoothing model for each individual customer 
	def update_exp_smooth_params_return_forecast_prices(self, comp_prices_last_t, t):
		for c in range(self.C-1):
			self.Base_value[t-1][c]=(self.alpha*comp_prices_last_t[c])+((1-self.alpha)*(self.Base_value[t-2][c]+self.Trend[t-2][c]))
			
			self.Trend[t-1][c]=(self.beta*(self.Base_value[t-1][c]-self.Base_value[t-2][c]))+((1-self.beta)*self.Trend[t-2][c])
			
			self.prices_next_t[c]=max(0, min(100,self.Base_value[t-2][c]+self.Trend[t-1][c]))
		return self.prices_next_t
		
	def sort(self, F):
		size=len(F)
		E=list(F)
		B=[0]*len(F)
		ind_ord=[0 for i in range(size)]
		for i in range(size):
			B[i]=i
		D=[0 for i in range(size)]
		for i in range(size):
			smallest=E[0]
			position=0
			for j in range(1,size-i):
				if E[j]<smallest:
					smallest=E[j]
					position=j
			D[i]=smallest
			E.pop(position)
			ind_ord[i]=B.pop(position)
		return [D,ind_ord]