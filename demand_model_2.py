import math
#from future import division
import numpy as np

class demand_model_2(object):

	C=2
	

	def __init__(self, competitors):
		self.C=competitors
	
	def winning_competitor(self, prices_this_t, np):#, parameterdump
		#The customer only sees a random subset of the competitors
		#and then chooses the cheapest regardless the price
		Competitor_subset=list(range(self.C))
		
		#
		subset_size=int(round(np.random.uniform(0,1)*self.C))
		
		#print(subset_size)
		
		selected_comp_index=-1
		if subset_size>0:
			for i in range(self.C-subset_size):
				pop_ind=int(math.floor(np.random.uniform(0,1)*(self.C-i)))
				Competitor_subset.pop(pop_ind)#jim=
				#print(Competitor_subset,', ',i,', ',pop_ind,', ',(self.C-1-i),', ',jim)
			
			#find the lowest price competitor of these
			min_price=10000000000
			
			for comp_index in Competitor_subset:
				if prices_this_t[comp_index]<min_price:
					min_price=prices_this_t[comp_index]
					selected_comp_index=comp_index
		
		return selected_comp_index