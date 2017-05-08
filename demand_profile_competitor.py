from Competitor import Competitor
import math as Math
import numpy as np

class demand_profile_competitor(Competitor):

	#store parameterdump here (until competition)
	parameters=3
	iterations=100
	parameters_over_time=[]
	initial_step_lengths=[]
	parameter_selection_distribution=[0.33, 0.66, 1]
	param_bounds=[[0,100],[-Math.pi/2, Math.pi/2],[0.000000000001,5]]
	t0Factor=0.1
	x=[]
	max_param_ranges=[100, Math.pi, 5]
	times_across_space=6;#using a linearly decreasing time step
	
	last_parameters=[]
	
	#estimated model parameters (these need to be updated to values consistent with previous observations)
	B=1
	#assumed normal distribution
	SIGMA=10
	MU=50
	
	N=1;
	
	#a second implementation of simulated annealing for the demand model parameters to previous observations  
	#mu=[0,100], signma=[0, 20], B=[0,5], N=[free parameter, multiplicative step length] (for a given model the N that fits best can be directly calculated)
	parameters_2=3
	iterations_2=10
	initial_step_lengths_2=[]
	parameter_selection_distribution_2=[0.33, 0.66, 1]
	param_bounds_2=[[0,100],[0, 20],[0.000000000001,5]]
	t0Factor_2=0.1
	max_param_ranges_2=[100, 50, 5]
	times_across_space_2=6;#using a linearly decreasing time step
	
	#
	wtp_sample_size=10
	
	expected_demand=0
	
	np=None
	
	
	

	def __init__(self, competitor_number, np):
		Competitor.__init__(self, competitor_number)
		
		self.competitor_number=competitor_number
		self.np=np
		self.last_parameters=[0 for i in range(self.parameters)]
		#
		self.parameters_over_time=np.zeros((self.parameters, self.T))
		
		#
		unitarySumOfDecreasingStepLengths=0
		for i in range(self.iterations):
			unitarySumOfDecreasingStepLengths=unitarySumOfDecreasingStepLengths+(1-(i/self.iterations))
			
		#
		self.initial_step_lengths=np.zeros((self.parameters))
		for i in range(self.parameters):
			self.initial_step_lengths[i]=(self.times_across_space*self.max_param_ranges[i])/(self.parameter_selection_distribution[i]*unitarySumOfDecreasingStepLengths)
		
		self.initial_step_lengths_2=np.zeros((self.parameters_2))
		for i in range(self.parameters_2):
			self.initial_step_lengths_2[i]=(self.times_across_space_2*self.max_param_ranges_2[i])/(self.parameter_selection_distribution_2[i]*unitarySumOfDecreasingStepLengths)
	
	def p2(self, prices_historical, demand_historical, t):#, parameterdump
	
		# if it's the first day
		#if demand_historical.size == 0:
		if t == 0:
			#store the number of competitors parameter in parameterdump
			self.C=len(prices_historical)
			#
			#
			self.x=np.zeros((self.C))
			for i in range(self.C):
				self.x[i]=i/(self.C-1)
			
			#random initial price
			popt = np.random.uniform(0,100)
		elif t<2:
			#random initial price
			popt = np.random.uniform(0,100)
		else:
			#update demand model parameters (correction based on observations)
			
			#THE CURRENT DEMAND PROFILE FITTING MODEL IS A SEVERE BOTTLE NECK
			#MAYBE MULTI-ARMED BANDITS CAN BE USED TO SEARCH THE DEMAND MODEL SPACE
			#initial_parameters_2=[self.MU, self.SIGMA, self.B, self.N]#current ones are used as the initial solution
			#updated_params_2=self.simulatedAnnealing_2(self.iterations_2, self.parameters_2, initial_parameters_2, self.initial_step_lengths_2, self.parameter_selection_distribution_2, self.param_bounds_2, self.t0Factor_2, prices_historical, demand_historical, t, self.np)
			#self.MU, self.SIGMA, self.B, self.N=updated_params_2
			
			
			#prices offered in the previous time period
			sorted_prices=self.sort(prices_historical[:,t-1])
			
			#
			initial_parameters=[sorted_prices[0], Math.atan(sorted_prices[self.C-1]-sorted_prices[0]), 1]
			
			#find the parameters (of price profile=a+b*relative_competitor^c)
			self.parameters_over_time[:,t-1]=self.simulatedAnnealing(self.iterations, self.parameters, initial_parameters, self.initial_step_lengths, self.parameter_selection_distribution, self.param_bounds, self.t0Factor, self.x, sorted_prices, self.np)
			
			#fit parameter forecast model (previous 10 or so)
			A=np.zeros((t-1,2))
			#print(parameters_over_time[0][0:t])#slicing test
			A[:,0]=np.arange(0,t-1)#parameters_over_time[0][0:t]
			A[:,1]=np.ones((t-1))
			#a
			Y=self.parameters_over_time[0][0:t-1]
			w=np.linalg.lstsq(A,Y)[0]
			forecast_params=[t*w[0]+w[1]]
			#b
			Y=self.parameters_over_time[1][0:t-1]
			w=np.linalg.lstsq(A,Y)[0]
			forecast_params.append(t*w[0]+w[1])
			#c
			Y=self.parameters_over_time[2][0:t-1]
			w=np.linalg.lstsq(A,Y)[0]
			forecast_params.append(t*w[0]+w[1])
			
			
			#sample wtp and derive competitor probability distribution
			#find optimum d*p
			popt = self.next_price(forecast_params)
		
		return popt
	
	#Use linear regression with only the previous 50 data points	
	def p(self, prices_historical, demand_historical, t):#, parameterdump
		
		prev_time_periods_used_in_linear_regression=50
		PPU=prev_time_periods_used_in_linear_regression
		
		# if it's the first day
		#if demand_historical.size == 0:
		if t == 0:
			#store the number of competitors parameter in parameterdump
			self.C=len(prices_historical)
			#
			#
			self.x=np.zeros((self.C))
			for i in range(self.C):
				self.x[i]=i/(self.C-1)
			
			#random initial price
			popt = np.random.uniform(0,100)
		elif t<2:
			#random initial price
			popt = np.random.uniform(0,100)
		else:
			#update demand model parameters (correction based on observations)
			
			######################################################
			#THE CURRENT DEMAND PROFILE FITTING MODEL IS A SEVERE BOTTLE NECK
			#MAYBE MULTI-ARMED BANDITS CAN BE USED TO SEARCH THE DEMAND MODEL SPACE
			#initial_parameters_2=[self.MU, self.SIGMA, self.B, self.N]
			
			#current ones are used as the initial solution
			#updated_params_2=self.simulatedAnnealing_2(self.iterations_2, self.parameters_2, initial_parameters_2, self.initial_step_lengths_2, self.parameter_selection_distribution_2, self.param_bounds_2, self.t0Factor_2, prices_historical, demand_historical, t, self.np)
			
			#self.MU, self.SIGMA, self.B, self.N=updated_params_2
			#####################################################
			
			#prices offered in the previous time period
			sorted_prices=self.sort(prices_historical[:,t-1])
			
			#
			initial_parameters=[sorted_prices[0], Math.atan(sorted_prices[self.C-1]-sorted_prices[0]), 1]
			
			#find the parameters (of price profile=a+b*relative_competitor^c)
			self.parameters_over_time[:,t-1]=self.simulatedAnnealing(self.iterations, self.parameters, initial_parameters, self.initial_step_lengths, self.parameter_selection_distribution, self.param_bounds, self.t0Factor, self.x, sorted_prices, self.np)
			
			#fit parameter forecast model (previous 10 or so)
			time_periods=min(t-1,PPU)
			t_first=max(0, t-1-PPU)
			t_last=max(0, t-1)
			#A=np.zeros((t-1,2))
			A=np.zeros((time_periods,2))
			#print(parameters_over_time[0][0:t])#slicing test
			A[:,0]=np.arange(0,time_periods)#t-1parameters_over_time[0][0:t]
			A[:,1]=np.ones((time_periods))#t-1
			#a
			Y=self.parameters_over_time[0][t_first:t_last]
			w=np.linalg.lstsq(A,Y)[0]
			forecast_params=[(time_periods+1)*w[0]+w[1]]
			#b
			Y=self.parameters_over_time[1][t_first:t_last]
			w=np.linalg.lstsq(A,Y)[0]
			forecast_params.append((time_periods+1)*w[0]+w[1])
			#c
			Y=self.parameters_over_time[2][t_first:t_last]
			w=np.linalg.lstsq(A,Y)[0]
			forecast_params.append((time_periods+1)*w[0]+w[1])
			
			
			#sample wtp and derive competitor probability distribution
			#find optimum d*p
			popt = self.next_price(forecast_params)
		
		return popt
		
	def simulatedAnnealing(self, iterations, parameters, initial_parameters, initial_step_lengths, parameter_selection_distribution, param_bounds, t0Factor, x, prices, np):
		current_solution=[0 for i in range(parameters)]
		best_solution=[0 for i in range(parameters)]
		neighbour_solution=[0 for i in range(parameters)]
		for i in range(parameters):
			current_solution[i]=initial_parameters[i]
			best_solution[i]=initial_parameters[i]
		
		
		current_obj=self.evaluateErrorModel1(initial_parameters, x, prices)
		best_obj=current_obj
		
		#timesNonImprovingSolAccepted=0;
		
		iteration=0
		
		#temp=(1-(double)iteration/iterations)*t0Factor*Math.abs(best_obj)
		TT=-1
		
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
			
			
			
			obj=self.evaluateErrorModel1(neighbour_solution, x, prices)
			#
			delta=obj-best_obj
			accept_new_solution=False
			#is this solution an improvement
			if delta<0:
				#accept the solution
				accept_new_solution=True
			else:
				#accept with temperature dependent probability
				if np.random.uniform(0,1,1)<Math.exp(-delta/temp) or (obj-current_obj<=0):
					accept_new_solution=True
					#timesNonImprovingSolAccepted=timesNonImprovingSolAccepted+1
			
			#
			if accept_new_solution:
				current_obj=obj
				#if the solution that has just been evaluated is accepted
				for i in range(parameters):
					current_solution[i]=neighbour_solution[i]
				
				#
				if obj<best_obj:
					best_obj=obj
					for i in range(parameters):
						best_solution[i]=neighbour_solution[i]

			iteration=iteration+1
		#print("timesNonImprovingSolAccepted=",timesNonImprovingSolAccepted);

		return best_solution
	
	#initial parameters	
	def simulatedAnnealing_2(self, iterations, parameters, initial_parameters, initial_step_lengths, parameter_selection_distribution, param_bounds, t0Factor, prices_historical, demand_historical, t, np):
		current_solution=[0 for i in range(parameters+1)]
		best_solution=[0 for i in range(parameters+1)]
		neighbour_solution=[0 for i in range(parameters+1)]
		for i in range(parameters):
			current_solution[i]=initial_parameters[i]
			best_solution[i]=initial_parameters[i]
		
		# params, prices_historical, demand_historical, t
		[obj,n]=self.evaluateDemandModel1(initial_parameters, prices_historical, demand_historical, t)
		best_solution[3]=n
		current_obj=obj
		best_obj=current_obj
		
		#timesNonImprovingSolAccepted=0;
		
		iteration=0
		
		#temp=(1-(double)iteration/iterations)*t0Factor*Math.abs(best_obj)
		TT=-1
		
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
				if np.random.uniform(0,1,1)<Math.exp(-delta/temp) or (obj-current_obj<=0):
					accept_new_solution=True
					#timesNonImprovingSolAccepted=timesNonImprovingSolAccepted+1
			
			#
			if accept_new_solution:
				current_obj=obj
				#if the solution that has just been evaluated is accepted
				for i in range(parameters):
					current_solution[i]=neighbour_solution[i]
				
				#
				if obj<best_obj:
					best_obj=obj
					for i in range(parameters):
						best_solution[i]=neighbour_solution[i]
					best_solution[3]=n
			iteration=iteration+1
		#print("timesNonImprovingSolAccepted=",timesNonImprovingSolAccepted);

		return best_solution
	
	def evaluateErrorModel1(self, params, x, prices):
		obj=0
		#
		for k in range(self.C):
			if x[k]==0:
				obj=obj+(params[0]-prices[k])**2
			else:
				obj=obj+((params[0]+Math.tan(params[1])*Math.pow(x[k],params[2]))-prices[k])**2
				
		return obj
		
	def evaluateDemandModel1(self, params, prices_historical, demand_historical, t):
		obj=0
		params[3]=0;
		#
		DM_TPs=50;
		t_first=max(0, t-DM_TPs-1)
		t_last=t-1
		
		#
		predicted_demands=[0 for i in range(t_first, t_last+1)]
		sum_of_predicted_demand=0
		sum_of_actual_demand=0
		for k in range(t_first, t_last+1):#
			sum_of_actual_demand=sum_of_actual_demand+demand_historical[k]
			
			#generate wtp smaple based on params
			wtp_sample_values=[]
			for i in range(self.wtp_sample_size):
				wtp_sample_values.append(self.quantileFunction(i/(self.wtp_sample_size-1), params[0], params[1]))

			
		
			#construct the demand distribution
			#print(prices_historical)
			#print(wtp_sample_values)
			sum=0
			demand_vector=[0 for i in range(self.C)]
			for i in range(self.C):#for each price
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
				
				#print(len(predicted_demands),',',k)
				predicted_demands[k-t_first]=demand_vector[self.competitor_number]
				sum_of_predicted_demand=sum_of_predicted_demand+predicted_demands[k-t_first]
				
		predicted_portion_of_demand=sum_of_predicted_demand/(t_last-t_first+1)
		if predicted_portion_of_demand>0:
			params[3]=sum_of_actual_demand/predicted_portion_of_demand
		

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
			B.pop(position)
		return D
	
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
		
	def next_price(self, forecast_params):
		#generate the set of predicted competitor prices 
		predicted_comp_prices=[]
		for i in range(self.C):
			predicted_comp_prices.append(forecast_params[0]+forecast_params[1]*(self.x[i]**forecast_params[2]))
			
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
		
		return next_price