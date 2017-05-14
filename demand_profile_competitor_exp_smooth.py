from Competitor import Competitor
import math as Math
import numpy as np

class demand_profile_competitor_exp_smooth(Competitor):

	Base_value=[]
	Trend=[]
	prices_next_t=[]
	alpha=0.2
	beta=0.2
	
	#estimated model parameters (these need to be updated to values consistent with previous observations)
	initB=2
	#assumed normal distribution
	initSIGMA=10
	initMU=50
	
	initN=1;
	
	B=2
	#assumed normal distribution
	SIGMA=10
	MU=50
	
	N=1;
	
	#a second implementation of simulated annealing for the demand model parameters to previous observations  
	#mu=[0,100], signma=[0, 20], B=[0,5], N=[free parameter, multiplicative step length] (for a given model the N that fits best can be directly calculated)
	parameters_2=3
	iterations_2=2
	initial_step_lengths_2=[]
	#parameter_selection_distribution_2=[0.33, 0.66, 1]
	parameter_selection_distribution_2=[0.33,0.66, 1]
	parameter_selection_distribution_2_step_len=[0.33,0.33, 0.33]
	param_bounds_2=[[0.000000000001,100],[0.000000000001, 50],[0.000000000001,5]]
	t0Factor_2=0.0000000001
	max_param_ranges_2=[100, 50, 5]
	times_across_space_2=0.0005;#using a linearly decreasing time step
	
	#
	wtp_sample_size=10
	
	expected_demand=0
	
	np=None
	
	
	

	def __init__(self, competitor_number, np):
		Competitor.__init__(self, competitor_number)
		
		self.competitor_number=competitor_number
		self.np=np
		
		
		
		unitarySumOfDecreasingStepLengths=0
		for i in range(self.iterations_2):
			unitarySumOfDecreasingStepLengths=unitarySumOfDecreasingStepLengths+(1-(i/self.iterations_2))
		
		self.initial_step_lengths_2=np.zeros((self.parameters_2))
		for i in range(self.parameters_2):
			self.initial_step_lengths_2[i]=(self.times_across_space_2*self.max_param_ranges_2[i])/(self.parameter_selection_distribution_2_step_len[i]*unitarySumOfDecreasingStepLengths)
			#self.initial_step_lengths_2[i]=0.01;
	
	#Use linear regression with only the previous 50 data points	
	def p(self, prices_historical, demand_historical, t):#, parameterdump
		
		prev_time_periods_used_in_linear_regression=50
		PPU=prev_time_periods_used_in_linear_regression
		
		# if it's the first day
		#if demand_historical.size == 0:
		if t == 0:
			#store the number of competitors parameter in parameterdump
			self.C=len(prices_historical)
			
			self.Base_value=[[0 for j in range(self.C)] for i in range(self.T)]
			self.Trend=[[0 for j in range(self.C)] for i in range(self.T)]
			
			self.prices_next_t=[0 for i in range(self.C)]
			
			self.MU=self.initMU
			self.SIGMA=self.initSIGMA
			self.B=self.initB
			self.N=self.initN
			
			#random initial price
			popt = np.random.uniform(0,100)
		elif t<2:
			#initialise base value (trend initialised to 0)
			#for c in range(self.C):
			[sorted_prices, ind_order]=self.sort(prices_historical[:,t-1])
			self.Base_value[0]=sorted_prices
			
			#random initial price
			popt = np.random.uniform(0,100)
		else:
			#update demand model parameters (correction based on observations)
			
			######################################################
			#THE CURRENT DEMAND PROFILE FITTING MODEL IS A SEVERE BOTTLE NECK
			#MAYBE MULTI-ARMED BANDITS CAN BE USED TO SEARCH THE DEMAND MODEL SPACE
			initial_parameters_2=[self.MU, self.SIGMA, self.B, self.N]
			
			#current ones are used as the initial solution
			updated_params_2=self.simulatedAnnealing_2(self.iterations_2, self.parameters_2, initial_parameters_2, self.initial_step_lengths_2, self.parameter_selection_distribution_2, self.param_bounds_2, self.t0Factor_2, prices_historical, demand_historical, t, self.np)
			
			[self.MU, self.SIGMA, self.B, self.N]=updated_params_2
			#####################################################
			
			#prices offered in the previous time period
			[sorted_prices, ind_order]=self.sort(prices_historical[:,t-1])
			
			
			forecast_price_set=self.update_exp_smooth_params_return_forecast_prices(sorted_prices, t)
			#sample wtp and derive competitor probability distribution
			#find optimum d*p
			#print(ind_order)
			us_index=0
			while ind_order[us_index]!=self.competitor_number:
				us_index=us_index+1
			#print(us_index)
			popt = self.next_price(forecast_price_set, us_index)
		
		return popt
		
	
	#initial parameters	
	def simulatedAnnealing_2(self, iterations, parameters, initial_parameters, initial_step_lengths, parameter_selection_distribution, param_bounds, t0Factor, prices_historical, demand_historical, t, np):
		current_solution=[0 for i in range(parameters+1)]
		best_solution=[0 for i in range(parameters+1)]
		neighbour_solution=[0 for i in range(parameters+1)]
		for i in range(parameters):
			current_solution[i]=initial_parameters[i]
			best_solution[i]=initial_parameters[i]
			neighbour_solution[i]=initial_parameters[i]
		
		# params, prices_historical, demand_historical, t
		[obj,n]=self.evaluateDemandModel1(neighbour_solution, prices_historical, demand_historical, t)
		best_solution[3]=n
		current_obj=obj
		best_obj=current_obj
		
		#
		
		#timesNonImprovingSolAccepted=0;
		
		iteration=0
		
		#temp=(1-(double)iteration/iterations)*t0Factor*Math.abs(best_obj)
		TT=-1
		
		#print(best_solution)
		
		while best_obj>0 and iteration<iterations:
			
			TT=iteration/iterations
			temp=(1-TT)*t0Factor*abs(best_obj)
			
			for i in range(parameters):
				neighbour_solution[i]=current_solution[i]
			
			
			#generate neighbouring solution
			rnd=np.random.uniform(0,1,1)
			param_to_modify=0
			while rnd>parameter_selection_distribution[param_to_modify]:
				param_to_modify=param_to_modify+1
			
			
			#pos/neg step
			if np.random.uniform(0,1,1)<0.5:
				neighbour_solution[param_to_modify]=min(param_bounds[param_to_modify][1], neighbour_solution[param_to_modify]+(initial_step_lengths[param_to_modify]*(1-TT)))
			else:
				neighbour_solution[param_to_modify]=max(param_bounds[param_to_modify][0], neighbour_solution[param_to_modify]-(initial_step_lengths[param_to_modify]*(1-TT)))
			
			#print(neighbour_solution)
			#evaluateDemandModel1
			[obj,n]=self.evaluateDemandModel1(neighbour_solution, prices_historical, demand_historical, t)
			#
			delta=obj-best_obj
			accept_new_solution=False
			#is this solution an improvement
			if delta<0:
				#accept the solution
				accept_new_solution=True
			else:
				#accept with temperature dependent probability
				p_accept=Math.exp(-delta/temp)
				rand_num=np.random.uniform(0,1,1)
				if rand_num<p_accept or (obj-current_obj<0):
					accept_new_solution=True
					#print(p_accept,',',rand_num,',',-delta/temp,',',TT,',',temp,',',current_obj,',',neighbour_solution)
					#timesNonImprovingSolAccepted=timesNonImprovingSolAccepted+1
			
			#
			if accept_new_solution:
				#print(obj,',',neighbour_solution)
				current_obj=obj
				#if the solution that has just been evaluated is accepted
				for i in range(parameters):
					current_solution[i]=neighbour_solution[i]
				current_solution[3]=n
				#
				if obj<best_obj:
					best_obj=obj
					for i in range(parameters):
						best_solution[i]=neighbour_solution[i]
					best_solution[3]=n
			iteration=iteration+1
		#print("timesNonImprovingSolAccepted=",timesNonImprovingSolAccepted);
		#print(best_solution)
		
		#
		
		return best_solution
	
		
	def evaluateDemandModel1(self, params, prices_historical, demand_historical, t):
		obj=0
		#params[3]=0;
		#
		DM_TPs=50;
		t_first=max(0, t-DM_TPs-1)
		t_last=t-1
		
		#
		#predicted_demands=[0 for i in range(t_first, t_last+1)]
		predicted_demands=[0 for i in range(t_last-t_first+1)]
		sum_of_predicted_demand=0
		sum_of_actual_demand=0
		non_zero_demand_periods=0
		#
		wtp_sample_values=[0 for i in range(self.wtp_sample_size)]
		for i in range(self.wtp_sample_size):
			wtp_sample_values[i]=self.quantileFunction(i/(self.wtp_sample_size-1), params[0], params[1])

		demand_vector=[0 for i in range(self.C)]
		for k in range(t_first, t_last+1):#
			sum_of_actual_demand=sum_of_actual_demand+demand_historical[k]
			
			#generate wtp smaple based on params
			
			
			
		
			#construct the demand distribution
			#print(prices_historical)
			#print(wtp_sample_values)
			sum=0
			#demand_vector=[0 for i in range(self.C)]
			for i in range(self.C):#for each price
				demand_vector[i]=0
				for j in range(self.wtp_sample_size):#for each sampled customer wtp
					#print(i," ",j)
					if prices_historical[i][k]<wtp_sample_values[j]:#then there is a change theat this customer will purchase from this competitor
						contribution=((wtp_sample_values[j]-prices_historical[i][k])/wtp_sample_values[j])**params[2]
						sum=sum+contribution
						demand_vector[i]=demand_vector[i]+contribution
			
			#correct the distribution so that it sums to 1
			if sum>0:
				for i in range(self.C):#for each price
					demand_vector[i]=demand_vector[i]/sum
				
				if demand_vector[self.competitor_number]>0:				
					non_zero_demand_periods=non_zero_demand_periods+1
				
				#print(len(predicted_demands),',',k)
				predicted_demands[k-t_first]=demand_vector[self.competitor_number]
				sum_of_predicted_demand=sum_of_predicted_demand+predicted_demands[k-t_first]
				
		predicted_portion_of_demand=sum_of_predicted_demand/(t_last-t_first)
		#non_zero_demand_periods#
		if predicted_portion_of_demand>0:
			params[3]=(sum_of_actual_demand/predicted_portion_of_demand)/(t_last-t_first)
			#print(demand_vector)params,',',
			#print(params[3],',',sum_of_actual_demand,',',predicted_portion_of_demand,',',sum_of_predicted_demand,',',self.C,',',(t_last-t_first))

		#multiple predicted demands by predicted N
		#and calculate demand squared error
		for k in range(t_first, t_last+1): 
			predicted_demands[k-t_first]=predicted_demands[k-t_first]*params[3]
			obj=obj+(demand_historical[k]-predicted_demands[k-t_first])**2
				
		return (obj, params[3])
		
	def sort(self, A):
		size=len(A)
		C=list(A)
		B=[0]*len(A)
		ind_ord=[0 for i in range(size)]
		for i in range(size):
			B[i]=i
		D=[0 for i in range(size)]
		for i in range(size):
			smallest=C[0]
			position=0
			for j in range(1,size-i):
				if C[j]<smallest:
					smallest=C[j]
					position=j
			D[i]=smallest
			C.pop(position)
			ind_ord[i]=B.pop(position)
		return [D,ind_ord]
	
	#normal distribution (equations from wikipedia)
	def inverseErrorFunctionApprox(self, z):
		result=0
		sign=1
		if z<0:
			sign=-1
		a=0.147
		result=sign*Math.sqrt((Math.sqrt(Math.pow((2/(Math.pi*a))+(Math.log(1-Math.pow(z, 2))/2), 2)-((Math.log(1-Math.pow(z, 2)))/(a)))-(((2)/(Math.pi*a))+(((Math.log(1-Math.pow(z, 2))))/(2)))))
		return result
	
	#normal distribution (equations from wikipedia)
	def quantileFunction(self, prob, mu, sigma):
		x=0
		if prob<=0:
			x=mu-sigma*2.5
		elif prob>=1:
			x=mu+sigma*2.5
		else:
			x=mu+sigma*Math.sqrt(2)*self.inverseErrorFunctionApprox(2*prob-1)
		return x
		
	#sorted prices could provide a more stable model, especially as competitor prices will not in general be modelled well with an exponential smoothing model for each individual customer 
	def update_exp_smooth_params_return_forecast_prices(self, comp_prices_last_t, t):
		for c in range(self.C):
			self.Base_value[t-1][c]=(self.alpha*comp_prices_last_t[c])+((1-self.alpha)*(self.Base_value[t-2][c]+self.Trend[t-2][c]))
			
			self.Trend[t-1][c]=(self.beta*(self.Base_value[t-1][c]-self.Base_value[t-2][c]))+((1-self.beta)*self.Trend[t-2][c])
			
			self.prices_next_t[c]=max(0, min(100,self.Base_value[t-2][c]+self.Trend[t-1][c]))
		return self.prices_next_t
		
	def next_price_2(self, predicted_comp_prices):
		#print(predicted_comp_prices)
		#equal probability interval sample
		wtp_sample_values=[]	
		for i in range(self.wtp_sample_size):
			wtp_sample_values.append(self.quantileFunction(i/(self.wtp_sample_size-1), self.MU, self.SIGMA))
			
		#generate a demand vector for each price from the wtp sample
		sum=0
		demand_vector=[0 for i in range(self.C)]
		for i in range(self.C):#for each price
			for j in range(self.wtp_sample_size):#for each sampled customer wtp
				if predicted_comp_prices[i]<wtp_sample_values[j]:#then there is a change theat this customer will purchase from this competitor
					contribution=((wtp_sample_values[j]-predicted_comp_prices[i])/wtp_sample_values[j])**self.B
					if wtp_sample_values[j]<=0:
						contribution=0
					sum=sum+contribution
					demand_vector[i]=demand_vector[i]+contribution
					
		#correct sum to 1
		next_price=self.np.random.uniform(0,100)
		#self.expected_demand=0 calculate this from observed demands for use in correcting model parameters. Below we are maximising demand magnitude without knowing the size of the population, but using the normal wtp and A*((wtp-price)/wtp)^B model. No attempt is made to estimate the size of the population (but surely this matters)
		if sum>0:
			for i in range(self.C):#for each price
				demand_vector[i]=demand_vector[i]/sum
				
			max_profit=0
			for i in range(self.C):#for each price
				prof=demand_vector[i]*predicted_comp_prices[i]
				if prof>max_profit:
					max_profit=prof
					next_price=predicted_comp_prices[i]
					
			#linear interpolation of intermediate prices and undercut and overcut prices
		
		return next_price

	def next_price(self, predicted_comp_prices, us_index):
		#print(predicted_comp_prices)
		#equal probability interval sample
		wtp_sample_values=[]	
		for i in range(self.wtp_sample_size):
			wtp_sample_values.append(self.quantileFunction(i/(self.wtp_sample_size-1), self.MU, self.SIGMA))
		
		next_price=self.np.random.uniform(0,100)
		
		#best_price=self.np.random.uniform(0,100)
		best_profit=0
		for c in range(self.C):
			if c!=us_index:
				#copy price c (this case specifically evaluates the profits associated with copying competitor c predicted price)
				#generate a demand vector for each price from the wtp sample
				sum=0
				demand_vector=[0 for i in range(self.C)]
				for i in range(self.C):#for each price
					if i!=us_index:
						for j in range(self.wtp_sample_size):#for each sampled customer wtp
							if predicted_comp_prices[i]<wtp_sample_values[j]:#then there is a change theat this customer will purchase from this competitor
								contribution=((wtp_sample_values[j]-predicted_comp_prices[i])/wtp_sample_values[j])**self.B
								if wtp_sample_values[j]<=0:
									contribution=0
								sum=sum+contribution
								demand_vector[i]=demand_vector[i]+contribution

						if i==c:
							#two competitors will be offering this price (one of which is us)
							#just one competitor offering this price (not us)
							for j in range(self.wtp_sample_size):#for each sampled customer wtp
								if predicted_comp_prices[c]<wtp_sample_values[j]:#then there is a change theat this customer will purchase from this competitor
									contribution=((wtp_sample_values[j]-predicted_comp_prices[c])/wtp_sample_values[j])**self.B
									if wtp_sample_values[j]<=0:
										contribution=0
									sum=sum+contribution
									demand_vector[us_index]=demand_vector[us_index]+contribution
							
					
							
				#correct sum to 1
				#self.expected_demand=0 calculate this from observed demands for use in correcting model parameters. Below we are maximising demand magnitude without knowing the size of the population, but using the normal wtp and A*((wtp-price)/wtp)^B model. No attempt is made to estimate the size of the population (but surely this matters)
				if sum>0:
					for i in range(self.C):#for each price
						demand_vector[i]=demand_vector[i]/sum
						
					prof=demand_vector[us_index]*predicted_comp_prices[c]
					if prof>best_profit:
						best_profit=prof
						next_price=predicted_comp_prices[c]
			else:
				#the profit associated with charging our own predicted price
				#generate a demand vector for each price from the wtp sample
				sum=0
				demand_vector=[0 for i in range(self.C)]
				for i in range(self.C):#for each price
					for j in range(self.wtp_sample_size):#for each sampled customer wtp
						if predicted_comp_prices[i]<wtp_sample_values[j]:#then there is a change theat this customer will purchase from this competitor
							contribution=((wtp_sample_values[j]-predicted_comp_prices[i])/wtp_sample_values[j])**self.B
							if wtp_sample_values[j]<=0:
								contribution=0
							sum=sum+contribution
							demand_vector[i]=demand_vector[i]+contribution
							
				#self.expected_demand=0 calculate this from observed demands for use in correcting model parameters. Below we are maximising demand magnitude without knowing the size of the population, but using the normal wtp and A*((wtp-price)/wtp)^B model. No attempt is made to estimate the size of the population (but surely this matters)
				if sum>0:
					for i in range(self.C):#for each price
						demand_vector[i]=demand_vector[i]/sum
						

					prof=demand_vector[c]*predicted_comp_prices[c]
					if prof>best_profit:
						best_profit=prof
						next_price=predicted_comp_prices[c]
						
				#linear interpolation of intermediate prices and undercut and overcut prices
		
		return next_price	