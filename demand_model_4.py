import math
#from future import division
import numpy as np

class demand_model_4(object):

	C=2
	a_1=1
	b_1=1
	c_1=1
	d_1=1
	e_1=1
	
	#for the case of e set to 1 linear regression can be used to fit the model

	def __init__(self, competitors, a_1, b_1, c_1, d_1, e_1):
		self.C=competitors
		self.a_1=a_1
		self.b_1=b_1
		self.c_1=c_1
		self.d_1=d_1
		self.e_1=e_1
	
	def winning_competitor(self, prices_this_t, np):#, parameterdump
		#The customer only sees a random subset of the competitors
		#and then chooses the cheapest regardless the price
		Competitor_subset=list(range(self.C))
		
		#
		subset_size=self.a+round(np.random.uniform(0,1)*(self.b-self.a))
		
		selected_comp_index=-1
		if subset_size>0:
			for i in range(self.C-subset_size):
				Competitor_subset.pop(math.floor(np.random.uniform(0,1)*subset_size))
			
			#find the lowest price competitor of these
			min_price=10000000000
			
			for comp_index in Competitor_subset:
				if prices_this_t[comp_index]<min_price:
					min_price=prices_this_t[comp_index]
					selected_comp_index=comp_index
		
		return selected_comp_index
		
	def mean_ignoring_one(self, prices_this_t, ignore_index, np):#, parameterdump
		mean=0
		for i in range(len(prices_this_t)):
			if i!=ignore_index:
				mean=mean+prices_this_t[i]
		mean=mean/(len(prices_this_t)-1)
		return mean
		
	def sd_ignoring_one(self, prices_this_t, ignore_index, np):#, parameterdump
		mean=mean_ignoring_one(prices_this_t, ignore_index, np)
		SS=0;
		number_of_values=len(prices_this_t)
		if number_of_values>2
			for i in range(len(prices_this_t)):
				if i!=ignore_index:
					SS=SS+((prices_this_t[i]-mean)**2);
			SS=SS/(len(prices_this_t)-2);
			SS=SS**0.5;
		return SS;