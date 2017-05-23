class demand_model_5(object):

	#model parameters
	a=1;
	b=1;

	#normal willingness to pay distribution
	mu=50
	sigma=10
	
	C=2
	
	cust_comp_select_cumu_dist=[]
	
	def __init__(self, competitors, a, b, mu, sigma):
		self.C=competitors
		self.a=a
		self.b=b
		self.mu=mu
		self.sigma=sigma
		self.cust_comp_select_cumu_dist=[0 for i in range(competitors)]
	
	def winning_competitor(self, prices_this_t, np):#, parameterdump
		#generate random willingness to pay
		cust_w_t_p=np.random.normal(self.mu, self.sigma, 1)
		#generate (cumulative) probability distribution for competitor selection (0 if price>= w.t.p)
		sum_cumu_dist=0
		for c in range(self.C):
			if prices_this_t[c]<cust_w_t_p:
				sum_cumu_dist=sum_cumu_dist+self.a*(1-((cust_w_t_p-prices_this_t[c])/cust_w_t_p))**self.b
				self.cust_comp_select_cumu_dist[c]=sum_cumu_dist
			else:
				self.cust_comp_select_cumu_dist[c]=sum_cumu_dist
				
		#make the probabilities sum to 1
		selected_comp_index=-1
		if sum_cumu_dist>0:
			for c in range(self.C):
				self.cust_comp_select_cumu_dist[c]=self.cust_comp_select_cumu_dist[c]/sum_cumu_dist
			#print(self.cust_comp_select_cumu_dist)
			
			#generate a random number to select a competitor to buy the product from
			rand_comp_selection_number=np.random.uniform(0,1,1)
			#print(rand_comp_selection_number)
			selected_comp_index=0
			while rand_comp_selection_number>=self.cust_comp_select_cumu_dist[selected_comp_index]:
				selected_comp_index=selected_comp_index+1
		
		return selected_comp_index