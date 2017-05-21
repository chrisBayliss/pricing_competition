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
	initB=1
	initSIGMA=20
	initMU=70
	initN=1
	
	B=2
	SIGMA=10
	MU=50
	N=1
	
	PPU=50
	
	WTPSS=25
	ZScores=[-2.0525351784516745,-1.5547210264744065,-1.2817128991385738,-1.0805041114062115,-0.9155202196139449,-0.7723070825166194,-0.6434209650126668,-0.5244456523403735,-0.4124867055596822,-0.30549085942962084,-0.2018964860015789,-0.10043409790833129,0.0,0.10043409790833129,0.2018964860015789,0.30549085942962084,0.4124867055596822,0.5244456523403735,0.6434209650126668,0.7723070825166194,0.9155202196139449,1.0805041114062115,1.2817128991385738,1.5547210264744071,2.0525351784516745,]
	#WTPSS=100
	#ZScores=[-2.5724655387049906,-2.1684599309842865,-1.9590489380232072,-1.811392700417685,-1.6951255314671452,-1.5980814841640125,-1.5140978937571379,-1.4396003356430906,-1.372321668169808,-1.3107291349449381,-1.2537355211790808,-1.2005401847480213,-1.1505354841879978,-1.1032486849492416,-1.058304292774104,-1.015398735127869,-0.9742828165154709,-0.9347492395472065,-0.8966235262394615,-0.8597572811063086,-0.8240231038913786,-0.7893106877951082,-0.7555237849300253,-0.7225778163747846,-0.6903979682827458,-0.6589176592955888,-0.6280772949785438,-0.5978232465333634,-0.5681070065006705,-0.5388844854043503,-0.5101154215692982,-0.4817628825109444,-0.4537928409385943,-0.42617381194541953,-0.39887654066729156,-0.371873731789539,-0.3451398139158496,-0.3186507330984484,-0.2923837708445634,-0.2663173827224585,-0.24043105433669845,-0.21470517196169633,-0.18912090554128483,-0.1636601021005565,-0.13830518789028187,-0.113039077805617,-0.08784509079950176,-0.06270687015391799,-0.03760830758515286,-0.01253347024610061,0.01253347024610061,0.03760830758515286,0.06270687015391799,0.08784509079950176,0.113039077805617,0.13830518789028187,0.1636601021005565,0.18912090554128483,0.21470517196169633,0.24043105433669845,0.2663173827224585,0.2923837708445634,0.3186507330984484,0.3451398139158496,0.371873731789539,0.39887654066729156,0.42617381194541953,0.4537928409385943,0.4817628825109444,0.5101154215692982,0.5388844854043503,0.5681070065006705,0.5978232465333634,0.6280772949785438,0.6589176592955888,0.6903979682827458,0.7225778163747846,0.7555237849300253,0.7893106877951082,0.8240231038913786,0.8597572811063086,0.8966235262394625,0.9347492395472065,0.9742828165154709,1.015398735127869,1.058304292774104,1.1032486849492416,1.1505354841879978,1.2005401847480213,1.2537355211790808,1.3107291349449381,1.3723216681698085,1.439600335643091,1.5140978937571383,1.5980814841640127,1.6951255314671456,1.8113927004176849,1.9590489380232072,2.1684599309842865,2.5724655387049906]
	
	#a second implementation of simulated annealing for the demand model parameters to previous observations  
	#mu=[0,100], signma=[0, 20], B=[0,5], N=[free parameter, multiplicative step length] (for a given model the N that fits best can be directly calculated)
	parameters_2=3
	iterations_2=2
	initial_step_lengths_2=[]
	#parameter_selection_distribution_2=[0.33, 0.66, 1]
	parameter_selection_distribution_2=[0.33,0.66, 1]
	parameter_selection_distribution_2_step_len=[0.33,0.33, 0.33]
	param_bounds_2=[[0.000000000001,100],[1, 30],[0.1,5]]
	t0Factor_2=0.001
	max_param_ranges_2=[100, 29, 4.9]
	times_across_space_2=6;#using a linearly decreasing time step
	
	#
	wtp_sample_size=10
	
	expected_demand=0
	
	np=None
	
	
	

	def __init__(self, competitor_number, np):
		Competitor.__init__(self, competitor_number)
		
		self.competitor_number=competitor_number
		self.np=np
		
		
		
		unitarySumOfDecreasingStepLengths=0
		for i in range(self.iterations_2*1000):
			unitarySumOfDecreasingStepLengths=unitarySumOfDecreasingStepLengths+(1-(i/(self.iterations_2*1000)))
		
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
			
			it_num=(t*iterations)+iteration
			TT=it_num/(iterations*1000)
			#TT=iteration/iterations
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
		pIntSize=1/self.WTPSS
		
		#
		#predicted_demands=[0 for i in range(t_first, t_last+1)]
		predicted_demands=[0 for i in range(t_last-t_first+1)]
		
		#sum_of_predicted_profit=0
		#sum_of_actual_profit=0
		#non_zero_demand_periods=0
		
		##generate wtp smaple based on params
		wtpVals=[0 for i in range(self.WTPSS)]
		for i in range(self.WTPSS):
			wtpVals[i]=params[0]+self.ZScores[i]*params[1]
		
		NEst=0
		
		demand_vector=[0 for i in range(self.C)]
		for k in range(t_first, t_last+1):#
			#chance_of_sale=0
			for j in range(self.WTPSS):
				#construct the demand distribution
				#print(prices_historical)
				#print(wtpVals)
				sum=0
				#demand_vector=[0 for i in range(self.C)]
				for i in range(self.C):#for each price
					demand_vector[i]=0
					#for each sampled customer wtp
					#print(i," ",j)
					if prices_historical[i][k]<wtpVals[j]:#then there is a change theat this customer will purchase from this competitor
						contribution=((wtpVals[j]-prices_historical[i][k])/wtpVals[j])**params[2]
						sum=sum+contribution
						demand_vector[i]=demand_vector[i]+contribution
			
				#correct the distribution so that it sums to 1
				if sum>0:
					#chance_of_sale=chance_of_sale+pIntSize
					for i in range(self.C):#for each price
						demand_vector[i]=demand_vector[i]/sum
				
				
				#print(len(predicted_demands),',',k)
				predicted_demands[k-t_first]=predicted_demands[k-t_first]+(pIntSize*demand_vector[self.competitor_number])
				
			if predicted_demands[k-t_first]>0:
				#account for the chance of no sales
				#predicted_demands[k-t_first]=predicted_demands[k-t_first]*chance_of_sale
				NEst=NEst+(demand_historical[k]/predicted_demands[k-t_first])
		
		#average N
		NEst=NEst/(t_last-t_first)
		params[3]=NEst
		
		#error on predicted profits 
		for k in range(t_first, t_last+1): 
			predicted_demands[k-t_first]=predicted_demands[k-t_first]*params[3]
			obj=obj+((demand_historical[k]*-prices_historical[self.competitor_number][k])-(predicted_demands[k-t_first]*NEst*prices_historical[self.competitor_number][k]))**2
				
		return (obj, params[3])
		
	def evaluateDemandModel1ZZZZZ(self, params, prices_historical, demand_historical, t):
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
		
	def sort(self, AA):
		size=len(AA)
		EE=list(AA)
		BB=[0]*len(AA)
		ind_ord=[0 for i in range(size)]
		for i in range(size):
			BB[i]=i
		DD=[0 for i in range(size)]
		for i in range(size):
			smallest=EE[0]
			position=0
			for j in range(1,size-i):
				if EE[j]<smallest:
					smallest=EE[j]
					position=j
			DD[i]=smallest
			EE.pop(position)
			ind_ord[i]=BB.pop(position)
		return [DD,ind_ord]
	
	#normal distribution (equations from wikipedia)
	def inverseErrorFunctionApprox(self, z):
		result=0
		sign=1
		if z<0:
			sign=-1
		aaa=0.147
		result=sign*Math.sqrt((Math.sqrt(Math.pow((2/(Math.pi*aaa))+(Math.log(1-Math.pow(z, 2))/2), 2)-((Math.log(1-Math.pow(z, 2)))/(aaa)))-(((2)/(Math.pi*aaa))+(((Math.log(1-Math.pow(z, 2))))/(2)))))
		return result
	
	#normal distribution (equations from wikipedia)
	def quantileFunction(self, prob, mu, sigma):
		x=0
		if prob<=0:
			x=mu-sigma*2.5
		elif prob>=1:
			x=mu+sigma*2.5
		else:
			x=mu+sigma*min(2.5, max(-2.5, Math.sqrt(2)*self.inverseErrorFunctionApprox(2*prob-1)))
		return x
	
	def CDFApprox(self, x):
		return 0.5*(1+errorFunctionApprox(x/(2**0.5)))
	
	def errorFunctionApprox(self, z):
		sign=1;
		if z<0:
			sign=-1
		aaa=0.147
		return sign*((1-Math.exp(-(z**2)*(((4/Math.pi)+(aaa*(z**2)))/(1+(aaa*(z**2))))))**0.5)
	
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