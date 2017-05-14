from Competitor import Competitor
import math as Math
import numpy as np

class demand_profile_competitor_cheapest_DM_exp_smooth(Competitor):

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
	
	#the cheapest subset demand model
	A_3=1
	initA_3=1
	B_3=1
	initB_3=1
	
	
	N=1;#common to all demand models
	
	demand_model_to_use=1
	
	
	#remember to set parameter settings based on the number of competitors
	
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
	
	parameters_3=2
	iterations_3=10
	initial_step_lengths_3=[]
	#parameter_selection_distribution_2=[0.33, 0.66, 1]
	parameter_selection_distribution_3=[0.5,1]
	parameter_selection_distribution_3_step_len=[0.5,0.5]
	param_bounds_3=[[0,100],[0,100]]
	t0Factor_3=0.0000000001
	max_param_ranges_3=[100, 100]
	times_across_space_3=0.00000002;#using a linearly decreasing time step
	
	#
	wtp_sample_size=10
	
	expected_demand=0
	
	other_prices=[]
	
	np=None
	
	PPU=50
	

	def __init__(self, competitor_number, np):
		Competitor.__init__(self, competitor_number)
		
		self.competitor_number=competitor_number
		self.np=np
		
		
		#wtp demand model
		unitarySumOfDecreasingStepLengths=0
		for i in range(self.iterations_2):
			unitarySumOfDecreasingStepLengths=unitarySumOfDecreasingStepLengths+(1-(i/self.iterations_2))
		
		self.initial_step_lengths_2=np.zeros((self.parameters_2))
		for i in range(self.parameters_2):
			self.initial_step_lengths_2[i]=(self.times_across_space_2*self.max_param_ranges_2[i])/(self.parameter_selection_distribution_2_step_len[i]*unitarySumOfDecreasingStepLengths)
			#self.initial_step_lengths_2[i]=0.01;
		
		#cheapest of subset demand model	
		unitarySumOfDecreasingStepLengths=0
		for i in range(self.iterations_3):
			unitarySumOfDecreasingStepLengths=unitarySumOfDecreasingStepLengths+(1-(i/self.iterations_3))
		
		self.initial_step_lengths_3=np.zeros((self.parameters_3))
		for i in range(self.parameters_3):
			self.initial_step_lengths_3[i]=(self.times_across_space_3*self.max_param_ranges_3[i])/(self.parameter_selection_distribution_3_step_len[i]*unitarySumOfDecreasingStepLengths)
			#self.initial_step_lengths_2[i]=0.01;
	
	#Use linear regression with only the previous 50 data points	
	def p(self, prices_historical, demand_historical, t):#, parameterdump
		
		
		
		# if it's the first day
		#if demand_historical.size == 0:
		if t == 0:
			#store the number of competitors parameter in parameterdump
			self.C=len(prices_historical)
			
			self.Base_value=[[0 for j in range(self.C-1)] for i in range(self.T)]
			self.Trend=[[0 for j in range(self.C-1)] for i in range(self.T)]
			
			self.prices_next_t=[0 for i in range(self.C-1)]
			
			self.MU=self.initMU
			self.SIGMA=self.initSIGMA
			self.B=self.initB
			self.N=self.initN
			
			#cheapest of subset demand model
			#set a<=b<=C parameter bounds
			self.initB_3=max(1,round(self.C))
			self.initA_3=self.initB_3=round(self.initB_3/2)
			
			self.A_3=self.initA_3
			self.B_3=self.initB_3
			
			self.other_prices=[0 for i in range(self.C-1)]
			
			#random initial price
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
			#update demand model parameters (correction based on observations)
			
			
			if self.demand_model_to_use==1:
				#DEMAND MODEL 1
				######################################################
				#THE CURRENT DEMAND PROFILE FITTING MODEL IS A SEVERE BOTTLE NECK
				#MAYBE MULTI-ARMED BANDITS CAN BE USED TO SEARCH THE DEMAND MODEL SPACE
				initial_parameters_2=[self.MU, self.SIGMA, self.B, self.N]
				
				#current ones are used as the initial solution
				updated_params_2=self.simulatedAnnealing_2(self.iterations_2, self.parameters_2, initial_parameters_2, self.initial_step_lengths_2, self.parameter_selection_distribution_2, self.param_bounds_2, self.t0Factor_2, prices_historical, demand_historical, t, self.np)
				
				[self.MU, self.SIGMA, self.B, self.N]=updated_params_2
				#####################################################
			elif self.demand_model_to_use==2:
				#DEMAND MODEL 2
				######################################################
				#MAYBE MULTI-ARMED BANDITS CAN BE USED TO SEARCH THE DEMAND MODEL SPACE
				initial_parameters_3=[self.A_3, self.B_3, self.N]
				
				#current ones are used as the initial solution
				updated_params_3=self.simulatedAnnealing_3(self.iterations_3, self.parameters_3, initial_parameters_3, self.initial_step_lengths_3, self.parameter_selection_distribution_3, self.param_bounds_3, self.t0Factor_3, prices_historical, demand_historical, t, self.np)
				
				[self.A_3, self.B_3, self.N]=updated_params_3
				#####################################################
			
			
			
			
			
			
			#demand model epsilon greedy is one to try
			
			
			
			#prices offered in the previous time period
			counter=0
			for i in range(self.C):
				if i!=self.competitor_number:
					self.other_prices[counter]=prices_historical[i,t-1]
					counter=counter+1
			
			[sorted_prices, ind_order]=self.sort(self.other_prices)#prices_historical[:,t-1]
			
			
			forecast_price_set=self.update_exp_smooth_params_return_forecast_prices(sorted_prices, t)
			#sample wtp and derive competitor probability distribution
			#find optimum d*p
			#print(ind_order)
			if self.demand_model_to_use==1:
				popt = self.next_price(forecast_price_set)
			elif self.demand_model_to_use==2:
				popt = self.next_price_demand_model_2(forecast_price_set)
		
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
		print(best_solution)
		
		#
		
		return best_solution
	
		
	def evaluateDemandModel1(self, params, prices_historical, demand_historical, t):
		obj=0
		#params[3]=0;
		#
		DM_TPs=self.PPU;
		t_first=max(0, t-DM_TPs-1)
		t_last=t-1
		
		#
		#predicted_demands=[0 for i in range(t_first, t_last+1)]
		predicted_demands=[0 for i in range(t_last-t_first+1)]
		sum_of_predicted_demand=0
		sum_of_actual_demand=0
		#sum_of_predicted_profit=0
		#sum_of_actual_profit=0
		#non_zero_demand_periods=0
		
		##generate wtp smaple based on params
		wtp_sample_values=[0 for i in range(self.wtp_sample_size)]
		for i in range(self.wtp_sample_size):
			wtp_sample_values[i]=self.quantileFunction(i/(self.wtp_sample_size-1), params[0], params[1])
		demand_vector=[0 for i in range(self.C)]
		for k in range(t_first, t_last+1):#
			sum_of_actual_demand=sum_of_actual_demand+demand_historical[k]
			
		
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
				
				#if demand_vector[self.competitor_number]>0:				
					#non_zero_demand_periods=non_zero_demand_periods+1
				
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
		
	#initial parameters	
	def simulatedAnnealing_3(self, iterations, parameters, initial_parameters, initial_step_lengths, parameter_selection_distribution, param_bounds, t0Factor, prices_historical, demand_historical, t, np):
		current_solution=[0 for i in range(parameters+1)]
		best_solution=[0 for i in range(parameters+1)]
		neighbour_solution=[0 for i in range(parameters+1)]
		for i in range(parameters):
			current_solution[i]=initial_parameters[i]
			best_solution[i]=initial_parameters[i]
			neighbour_solution[i]=initial_parameters[i]
		
		# params, prices_historical, demand_historical, t
		[obj,n]=self.evaluateDemandModel2(neighbour_solution, prices_historical, demand_historical, t)
		best_solution[2]=n
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
			pos_neg=1
			if np.random.uniform(0,1,1)<0.5:
				pos_neg=-1
			
			#print('param_to_modify=',param_to_modify,', pos_neg=',pos_neg,', neighbour_solution[0]=',neighbour_solution[0])
			
			if param_to_modify==0:#A_3
				
				neighbour_solution[0]=min(self.C, max(0, neighbour_solution[0]+pos_neg*(max(1, initial_step_lengths[0]))))
				neighbour_solution[1]=max(neighbour_solution[0], neighbour_solution[1])
				#print(neighbour_solution)
			else:
				#The following ensures B_3>=1
				neighbour_solution[1]=min(self.C, max(1, neighbour_solution[1]+pos_neg*(max(1, initial_step_lengths[1]))))
				neighbour_solution[0]=min(neighbour_solution[0], neighbour_solution[1])
			
			
			#
			
			#evaluateDemandModel1
			[obj,n]=self.evaluateDemandModel2(neighbour_solution, prices_historical, demand_historical, t)
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
				current_solution[2]=n
				#
				if obj<best_obj:
					best_obj=obj
					for i in range(parameters):
						best_solution[i]=neighbour_solution[i]
					best_solution[2]=n
			iteration=iteration+1
		#print("timesNonImprovingSolAccepted=",timesNonImprovingSolAccepted);
		print(best_solution)
		
		#
		
		return best_solution
		
	def evaluateDemandModel2(self, params, prices_historical, demand_historical, t):
		obj=0
		#params[3]=0;
		#
		DM_TPs=self.PPU;
		t_first=max(0, t-DM_TPs-1)
		t_last=t-1
		
		#
		#predicted_demands=[0 for i in range(t_first, t_last+1)]
		predicted_demands=[0 for i in range(t_last-t_first+1)]
		sum_of_predicted_demand=0
		sum_of_actual_demand=0
		#sum_of_predicted_profit=0
		#sum_of_actual_profit=0
		#non_zero_demand_periods=0
		
		
		demand_vector=[0 for i in range(self.C)]
		for k in range(t_first, t_last+1):#
			sum_of_actual_demand=sum_of_actual_demand+demand_historical[k]
			
			#our price's cheapness rank
			our_rank=1
			for i in range(self.C):
				if self.competitor_number!=i:
					if prices_historical[i][k]<prices_historical[self.competitor_number][k]:
						our_rank=our_rank+1
			
			#the calculations below COULD be speeded up by storing binomial coefficients
			#calculate the predicted demand portion given parameters a,b
			for r in range(max(1,params[0]), params[1]+1):
				#print('self.C',self.C,', our_rank=',our_rank,', r=',r)
				equationNumberOfWins=(self.factorial((self.C-our_rank))/(self.factorial(r-1)*self.factorial((self.C-our_rank)-(r-1))));
				
				out_of=self.factorial(self.C)/(self.factorial(r)*self.factorial(self.C-r))
				
				#if out_of==1:
					#print(params,', ',our_rank,', ',equationNumberOfWins,', ',out_of)
				#out_of>0 and 
				
				
				predicted_demands[k-t_first]=predicted_demands[k-t_first]+(1/(params[1]-params[0]+1))*(equationNumberOfWins/out_of)
				#if (params[1]-params[0])>0:
					#predicted_demands[k-t_first]=predicted_demands[k-t_first]+(1/(params[1]-params[0]))*(equationNumberOfWins/out_of)
				#elif (params[1]-params[0])==0:
					#predicted_demands[k-t_first]=predicted_demands[k-t_first]+(equationNumberOfWins/out_of)
				
			#sum of predicted demand portions
			sum_of_predicted_demand=sum_of_predicted_demand+predicted_demands[k-t_first]
			
			
				
		predicted_portion_of_demand=sum_of_predicted_demand/(t_last-t_first)
		#non_zero_demand_periods#
		if predicted_portion_of_demand>0:
			params[2]=(sum_of_actual_demand/(t_last-t_first))/predicted_portion_of_demand
			#print(demand_vector)params,',',
			#print(params[3],',',sum_of_actual_demand,',',predicted_portion_of_demand,',',sum_of_predicted_demand,',',self.C,',',(t_last-t_first))

		#multiple predicted demands by predicted N
		#and calculate demand squared error
		for k in range(t_first, t_last+1): 
			predicted_demands[k-t_first]=predicted_demands[k-t_first]*params[2]
			obj=obj+(demand_historical[k]-predicted_demands[k-t_first])**2
		
		return (obj, params[2])
		
	def factorial(self, value):
		result=1
		for i in range(value):
			result=result*(i+1)
		return result
		
	def sort(self, A):
		size=len(A)
		CC=list(A)
		B=[0]*len(A)
		ind_ord=[0 for i in range(size)]
		for i in range(size):
			B[i]=i
		D=[0 for i in range(size)]
		for i in range(size):
			smallest=CC[0]
			position=0
			for j in range(1,size-i):
				if CC[j]<smallest:
					smallest=CC[j]
					position=j
			D[i]=smallest
			CC.pop(position)
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
		for c in range(self.C-1):
			self.Base_value[t-1][c]=(self.alpha*comp_prices_last_t[c])+((1-self.alpha)*(self.Base_value[t-2][c]+self.Trend[t-2][c]))
			
			self.Trend[t-1][c]=(self.beta*(self.Base_value[t-1][c]-self.Base_value[t-2][c]))+((1-self.beta)*self.Trend[t-2][c])
			
			self.prices_next_t[c]=max(0, min(100,self.Base_value[t-2][c]+self.Trend[t-1][c]))
		return self.prices_next_t
		
	def next_price_ZZZZZ(self, predicted_comp_prices):
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

	def next_price(self, predicted_comp_prices):
		#print(predicted_comp_prices)
		#equal probability interval sample
		wtp_sample_values=[]	
		for i in range(self.wtp_sample_size):
			wtp_sample_values.append(self.quantileFunction(i/(self.wtp_sample_size-1), self.MU, self.SIGMA))
		
		next_price=self.np.random.uniform(0,100)
		
		#generate a set of potential prices. The build the demand vector for each of these
		#sort the forecast prices
		#+/-5 the most expensive, cheapest
		#each price and the mid points
		
		#sort the forecast competitor prices 
		[sorted_forecast_prices,ind_ord]=self.sort(predicted_comp_prices)
		
		potential_prices=[]
		potential_prices.append(sorted_forecast_prices[0]-5)
		potential_prices.append(sorted_forecast_prices[self.C-2]+5)
		for i in range(len(sorted_forecast_prices)):
			potential_prices.append(sorted_forecast_prices[i])
			if i<len(sorted_forecast_prices)-1:
				potential_prices.append((sorted_forecast_prices[i]+sorted_forecast_prices[i+1])/2)
		
		#build and evaluate the demand for each of these potential prices
		best_profit=-1
		for k in range(len(potential_prices)):
			#the profit associated with charging our own predicted price
			#generate a demand vector for each price from the wtp sample
			sum=0
			demand_vector=[0 for i in range(self.C)]
			for i in range(self.C-1):#for each price
				for j in range(self.wtp_sample_size):#for each sampled customer wtp
					if predicted_comp_prices[i]<wtp_sample_values[j]:#then there is a change theat this customer will purchase from this competitor
						contribution=((wtp_sample_values[j]-predicted_comp_prices[i])/wtp_sample_values[j])**self.B
						if wtp_sample_values[j]<=0:
							contribution=0
						sum=sum+contribution
						demand_vector[i]=demand_vector[i]+contribution
			
			#demand vector contribution from our price (we are index self.C-1)
			for j in range(self.wtp_sample_size):#for each sampled customer wtp
				if potential_prices[k]<wtp_sample_values[j]:#then there is a change theat this customer will purchase from this competitor
					contribution=((wtp_sample_values[j]-potential_prices[k])/wtp_sample_values[j])**self.B
					if wtp_sample_values[j]<=0:
						contribution=0
					sum=sum+contribution
					demand_vector[self.C-1]=demand_vector[self.C-1]+contribution
			
			#self.expected_demand=0 calculate this from observed demands for use in correcting model parameters. Below we are maximising demand magnitude without knowing the size of the population, but using the normal wtp and A*((wtp-price)/wtp)^B model. No attempt is made to estimate the size of the population (but surely this matters)
			if sum>0:
				for i in range(self.C):#for each price
					demand_vector[i]=demand_vector[i]/sum
					

				prof=demand_vector[self.C-1]*potential_prices[k]
				if prof>best_profit:
					best_profit=prof
					next_price=potential_prices[k]
		
		return next_price	
		
	def next_price_demand_model_2(self, predicted_comp_prices):
		#print(predicted_comp_prices)
		
		
		##the expected profit for undercutting each competitor by a bit and also charging a higher price (being ranked last, the probability only has to be calculated once) Highest price competitor plus 5 as well
		
		#find the demand portion associated with each rank
		rank_demand=[0]*self.C
		for i in range(self.C):
			for r in range(max(1,self.A_3), self.B_3+1):
				equationNumberOfWins=(self.factorial((self.C-(i+1)))/(self.factorial(r-1)*self.factorial((self.C-(i+1))-(r-1))));
				
				out_of=self.factorial(self.C)/(self.factorial(r)*self.factorial(self.C-r))
				
				rank_demand[i]=rank_demand[i]+(((1/(self.B_3-self.A_3+1))*(equationNumberOfWins/out_of))*self.N)
				#if self.B_3-self.A_3>0:
					#rank_demand[i]=rank_demand[i]+(((1/(self.B_3-self.A_3))*(equationNumberOfWins/out_of))*self.N)
				#elif self.B_3-self.A_3==0:
					#rank_demand[i]=rank_demand[i]+(((equationNumberOfWins/out_of))*self.N)
				
		
		#get the rank order of the prices 
		[sorted_forecast_prices,ind_ord]=self.sort(predicted_comp_prices)
		#print(rank_demand)
		#print(sorted_forecast_prices)
		
		#the expected profit for undercutting each competitor
		max_prof=-1
		next_price=self.np.random.uniform(0,100)
		for i in range(len(sorted_forecast_prices)):
			comp_price=sorted_forecast_prices[i]
			prof_of_undercut=(comp_price-1)*rank_demand[i]
			if prof_of_undercut>max_prof:
				max_prof=prof_of_undercut
				next_price=comp_price-1
		
		#the profit associated with being the most expensive
		#sorted_forecast_prices[self.C-2]+5
		prof_of_overcut=(100)*rank_demand[self.C-1]
		if prof_of_overcut>max_prof:
			max_prof=prof_of_overcut
			next_price=100#sorted_forecast_prices[self.C-2]+5
		
		#print(next_price)
		
		return next_price
		
	def reset(self):
		sel