import math as Math
import os, math, operator, random, csv, scipy, time
import numpy as np

#Use linear regression with only the previous 50 data points	
def p(prices_historical, demand_historical, parameterdump):#
	#declaration of inner functions
	###############################
	def update( chosen_arm, reward):
		#print(chosen_arm)
		counts_1[int(chosen_arm)] = counts_1[int(chosen_arm)] + 1
		n = counts_1[int(chosen_arm)]
		value = values_1[int(chosen_arm)]
		new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward #weighted average
		return new_value
	
	def update_2( chosen_arm, reward):
		#print(chosen_arm)
		counts_2[int(chosen_arm)] = counts_2[int(chosen_arm)] + 1
		n = counts_2[int(chosen_arm)]
		value = values_2[int(chosen_arm)]
		new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward #weighted average
		return new_value
		
	def select_arm( epsilon,values): #it returns which arm to play
		rand_num=np.random.uniform(0,1)
		#print(rand_num,',',epsilon)  
		if rand_num > epsilon: #it chooses randomly whether it will explore or exploit
			return ind_max(values) #it selects the price with the highest demand so far
		else:
			arm=random.randrange(len(values))
			#print(arm) 
			return arm #it selects a random arm
	def ind_max( x):
		m = max(x)
		return x.index(m)
	
	def simulated_annealing_all_demand_models( demand_model, iterations, parameters, initial_parameters, initial_step_lengths, parameter_selection_distribution, param_bounds, t0Factor, prices_historical, demand_historical, previous_non_zero_demand_time_periods, t, np):
		best_pprd=-1
		current_solution=[0 for i in range(parameters+1)]
		best_solution=[0 for i in range(parameters+1)]
		neighbour_solution=[0 for i in range(parameters+1)]
		for i in range(parameters):
			current_solution[i]=initial_parameters[i]
			best_solution[i]=initial_parameters[i]
			neighbour_solution[i]=initial_parameters[i]
		
		# params, prices_historical, demand_historical, t
		if demand_model==1:
			#wtp bargain hunters
			[obj,n,pprt]=evaluateDemandModel1(neighbour_solution, prices_historical, demand_historical, previous_non_zero_demand_time_periods, t)
		elif demand_model==2:
			#random subset
			[obj,n]=evaluateDemandModel2(neighbour_solution, prices_historical, demand_historical, previous_non_zero_demand_time_periods, t)
		else:#3
			#quality perceivers
			[obj,n]=evaluateDemandModel5(neighbour_solution, prices_historical, demand_historical, previous_non_zero_demand_time_periods, t)
		
		#current best
		best_solution[parameters]=n
		current_obj=obj
		best_obj=current_obj
		
		
		#timesNonImprovingSolAccepted=0;
		
		iteration=0
		
		#temp=(1-(double)iteration/iterations)*t0Factor*Math.abs(best_obj)
		TT=-1
		
		#print(best_solution)
		improvement_found=False
		
		while best_obj>0 and iteration<iterations:
			
			it_num=(t*iterations)+iteration
			TT=it_num/(iterations*1000)
			#TT=iteration/iterations
			temp=(1-TT)*t0Factor*abs(best_obj)
			
			#print(initial_step_lengths[0]*(1-TT))
			
			for i in range(parameters):
				neighbour_solution[i]=current_solution[i]
			

			#generate neighbouring solution
			rnd=np.random.uniform(0,1,1)
			param_to_modify=0
			while rnd>parameter_selection_distribution[param_to_modify]:
				param_to_modify=param_to_modify+1
			
			
			if demand_model==1 or demand_model==3:
				#pos/neg step (applicable for demand models 1 and 2)
				if np.random.uniform(0,1,1)<0.5:
					neighbour_solution[param_to_modify]=min(param_bounds[param_to_modify][1], neighbour_solution[param_to_modify]+(initial_step_lengths[param_to_modify]*(1-TT)))
				else:
					neighbour_solution[param_to_modify]=max(param_bounds[param_to_modify][0], neighbour_solution[param_to_modify]-(initial_step_lengths[param_to_modify]*(1-TT)))
			elif demand_model==2:#
				#pos/neg step	
				pos_neg=1
				if np.random.uniform(0,1,1)<0.5:
					pos_neg=-1

				if param_to_modify==0:#A_3
					
					neighbour_solution[0]=int(min(C, max(0, neighbour_solution[0]+pos_neg*1)))
					neighbour_solution[1]=int(max(neighbour_solution[0], neighbour_solution[1]))
					#print(neighbour_solution)
				else:
					#The following ensures B_3>=1
					neighbour_solution[1]=int(min(C, max(1, neighbour_solution[1]+pos_neg*1)))
					neighbour_solution[0]=int(min(neighbour_solution[0], neighbour_solution[1]))
			

			#evaluate the error
			if demand_model==1:
				#print(current_solution)
				#print(neighbour_solution)
				#wtp bargain hunters
				[obj,n,pprt]=evaluateDemandModel1(neighbour_solution, prices_historical, demand_historical, previous_non_zero_demand_time_periods, t)
			elif demand_model==2:
				#random subset
				[obj,n]=evaluateDemandModel2(neighbour_solution, prices_historical, demand_historical, previous_non_zero_demand_time_periods, t)
			else:#3
				#quality perceivers
				[obj,n]=evaluateDemandModel5(neighbour_solution, prices_historical, demand_historical, previous_non_zero_demand_time_periods, t)
			
				
			#print(obj,',',best_obj)
			
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
				current_solution[parameters]=n
				#
				if obj<best_obj:
					improvement_found=True
					best_obj=obj
					
					if demand_model==1 and fix_arm:
						best_pprd=pprt
					
					for i in range(parameters):
						best_solution[i]=neighbour_solution[i]
					best_solution[parameters]=n
			iteration=iteration+1
		#print("timesNonImprovingSolAccepted=",timesNonImprovingSolAccepted);
		
		#if demand_model==1 and fix_arm:
			#print(best_solution,', ',best_obj,', ',best_pprd)
			
		#apply sample size based correction to the sigma estimate
		#if improvement_found and (demand_model==1 or demand_model==3):
			#best_solution[1]=best_solution[1]*((best_solution[parameters]*best_solution[parameters+1])**0.5)
		
		return best_solution
		
	#model evaluation methods
	def evaluateDemandModel1( params, prices_historical, demand_historical, previous_non_zero_demand_time_periods, t):
		obj=0
		#params[3]=0;
		predicted_portion_of_demand=0
		
		#print(competitor_number)
		#
		DM_TPs=int(min(len(previous_non_zero_demand_time_periods), PPU))
		t_last=len(previous_non_zero_demand_time_periods)-1
		t_first=max(0, t_last-DM_TPs)#-1
		
		#print(t_first,', ',t_last)
		
		#
		#predicted_demands=[0 for i in range(t_first, t_last+1)]
		predicted_demands=[0 for i in range(t_last-t_first+1)]
		sum_of_predicted_demand=0
		sum_of_actual_demand=0
		#sum_of_predicted_profit=0
		#sum_of_actual_profit=0
		#non_zero_demand_periods=0
		
		##generate wtp smaple based on params
		wtp_sample_values=[0 for i in range(WTPSS)]
		for i in range(WTPSS):
			wtp_sample_values[i]=max(1,min(100,quantileFunction(i/(WTPSS-1), params[0], params[1])))
		demand_vector=[0 for i in range(C)]
		for k in range(t_first, t_last+1):#
			sum_of_actual_demand=sum_of_actual_demand+demand_historical[previous_non_zero_demand_time_periods[k]]
			
		
			#construct the demand distribution
			#print(prices_historical)
			#print(wtp_sample_values)
			sum=0
			#demand_vector=[0 for i in range(C)]
			for i in range(C):#for each price
				demand_vector[i]=0
				for j in range(WTPSS):#for each sampled customer wtp
					#print(i," ",j)
					if prices_historical[i][previous_non_zero_demand_time_periods[k]]<wtp_sample_values[j]:#then there is a change theat this customer will purchase from this competitor
						contribution=((wtp_sample_values[j]-prices_historical[i][previous_non_zero_demand_time_periods[k]])/wtp_sample_values[j])**params[2]
						sum=sum+contribution
						demand_vector[i]=demand_vector[i]+contribution
			
			#correct the distribution so that it sums to 1
			if sum>0:
				for i in range(C):#for each price
					demand_vector[i]=demand_vector[i]/sum
				
				#if demand_vector[competitor_number]>0:				
					#non_zero_demand_periods=non_zero_demand_periods+1
				
				#print(len(predicted_demands),',',k)
				predicted_demands[k-t_first]=demand_vector[competitor_number]
				sum_of_predicted_demand=sum_of_predicted_demand+predicted_demands[k-t_first]
		#print(sum_of_predicted_demand)
		#zero_predicteddemand=False
		if (t_last-t_first)>0:			
			predicted_portion_of_demand=sum_of_predicted_demand/(t_last-t_first)
			#non_zero_demand_periods#
			if predicted_portion_of_demand>0:
				params[3]=(sum_of_actual_demand/predicted_portion_of_demand)/(t_last-t_first)
				#print(demand_vector)params,',',
				#print(params[3],',',sum_of_actual_demand,',',predicted_portion_of_demand,',',sum_of_predicted_demand,',',C,',',(t_last-t_first))

			#multiple predicted demands by predicted N
			#and calculate demand squared error
			for k in range(t_first, t_last+1): 
				predicted_demands[k-t_first]=predicted_demands[k-t_first]*params[3]
				#obj=obj+(demand_historical[previous_non_zero_demand_time_periods[k]]-predicted_demands[k-t_first])**2
				#obj=obj+((demand_historical[previous_non_zero_demand_time_periods[k]]*prices_historical[competitor_number][previous_non_zero_demand_time_periods[k]])-(predicted_demands[k-t_first]*prices_historical[competitor_number][previous_non_zero_demand_time_periods[k]]))**2
				
				obj=obj+((demand_historical[previous_non_zero_demand_time_periods[k]]-predicted_demands[k-t_first])**2)+((0.01*((demand_historical[previous_non_zero_demand_time_periods[k]]*prices_historical[competitor_number][previous_non_zero_demand_time_periods[k]])-(predicted_demands[k-t_first]*prices_historical[competitor_number][previous_non_zero_demand_time_periods[k]])))**2)
				
			#else:
				#obj=10000000000000
		else:
			obj=10000000000000
		#print(predicted_portion_of_demand)		
		return (obj, params[3], predicted_portion_of_demand)
		
	def evaluateDemandModel5( params, prices_historical, demand_historical, previous_non_zero_demand_time_periods, t):
		obj=0
		#params[3]=0;
		#
		DM_TPs=int(min(len(previous_non_zero_demand_time_periods), PPU))
		t_last=len(previous_non_zero_demand_time_periods)-1
		t_first=max(0, t_last-DM_TPs)#-1
		
		#
		#predicted_demands=[0 for i in range(t_first, t_last+1)]
		predicted_demands=[0 for i in range(t_last-t_first+1)]
		sum_of_predicted_demand=0
		sum_of_actual_demand=0
		#sum_of_predicted_profit=0
		#sum_of_actual_profit=0
		#non_zero_demand_periods=0
		
		##generate wtp smaple based on params
		wtp_sample_values=[0 for i in range(WTPSS)]
		for i in range(WTPSS):
			wtp_sample_values[i]=max(1,min(100,quantileFunction(i/(WTPSS-1), params[0], params[1])))
		demand_vector=[0 for i in range(C)]
		for k in range(t_first, t_last+1):#
			sum_of_actual_demand=sum_of_actual_demand+demand_historical[previous_non_zero_demand_time_periods[k]]
			
		
			#construct the demand distribution
			#print(prices_historical)
			#print(wtp_sample_values)
			sum=0
			#demand_vector=[0 for i in range(C)]
			for i in range(C):#for each price
				demand_vector[i]=0
				for j in range(WTPSS):#for each sampled customer wtp
					#print(i," ",j)
					if prices_historical[i][previous_non_zero_demand_time_periods[k]]<wtp_sample_values[j]:#then there is a change theat this customer will purchase from this competitor
						contribution=max(0,(1-((wtp_sample_values[j]-prices_historical[i][previous_non_zero_demand_time_periods[k]])/wtp_sample_values[j])))**params[2]
						sum=sum+contribution
						demand_vector[i]=demand_vector[i]+contribution
			
			#correct the distribution so that it sums to 1
			if sum>0:
				for i in range(C):#for each price
					demand_vector[i]=demand_vector[i]/sum
				
				#if demand_vector[competitor_number]>0:				
					#non_zero_demand_periods=non_zero_demand_periods+1
				
				#print(len(predicted_demands),',',k)
				predicted_demands[k-t_first]=demand_vector[competitor_number]
				sum_of_predicted_demand=sum_of_predicted_demand+predicted_demands[k-t_first]
		
		if (t_last-t_first)>0:				
			predicted_portion_of_demand=sum_of_predicted_demand/(t_last-t_first)
			#non_zero_demand_periods#
			if predicted_portion_of_demand>0:
				params[3]=(sum_of_actual_demand/predicted_portion_of_demand)/(t_last-t_first)
				#print(demand_vector)params,',',
				#print(params[3],',',sum_of_actual_demand,',',predicted_portion_of_demand,',',sum_of_predicted_demand,',',C,',',(t_last-t_first))

				#multiple predicted demands by predicted N
				#and calculate demand squared error
			for k in range(t_first, t_last+1): 
				predicted_demands[k-t_first]=predicted_demands[k-t_first]*params[3]
				#obj=obj+(demand_historical[previous_non_zero_demand_time_periods[k]]-predicted_demands[k-t_first])**2
				#obj=obj+((demand_historical[previous_non_zero_demand_time_periods[k]]*prices_historical[competitor_number][previous_non_zero_demand_time_periods[k]])-(predicted_demands[k-t_first]*prices_historical[competitor_number][previous_non_zero_demand_time_periods[k]]))**2
				
				obj=obj+((demand_historical[previous_non_zero_demand_time_periods[k]]-predicted_demands[k-t_first])**2)+((0.01*((demand_historical[previous_non_zero_demand_time_periods[k]]*prices_historical[competitor_number][previous_non_zero_demand_time_periods[k]])-(predicted_demands[k-t_first]*prices_historical[competitor_number][previous_non_zero_demand_time_periods[k]])))**2)
				
			#else:
				#obj=1000000000000
		else:
			obj=1000000000000
				
		return (obj, params[3])
	
	def evaluateDemandModel2( params, prices_historical, demand_historical, previous_non_zero_demand_time_periods, t):
		obj=0
		#params[3]=0;
		#
		DM_TPs=int(min(len(previous_non_zero_demand_time_periods), PPU))
		t_last=len(previous_non_zero_demand_time_periods)-1
		t_first=max(0, t_last-DM_TPs)#-1
		
		#
		#predicted_demands=[0 for i in range(t_first, t_last+1)]
		predicted_demands=[0 for i in range(t_last-t_first+1)]
		sum_of_predicted_demand=0
		sum_of_actual_demand=0
		#sum_of_predicted_profit=0
		#sum_of_actual_profit=0
		#non_zero_demand_periods=0
		
		
		demand_vector=[0 for i in range(C)]
		for k in range(t_first, t_last+1):#
			sum_of_actual_demand=sum_of_actual_demand+demand_historical[previous_non_zero_demand_time_periods[k]]
			
			#our price's cheapness rank
			our_rank=1
			for i in range(C):
				if competitor_number!=i:
					if prices_historical[i][previous_non_zero_demand_time_periods[k]]<prices_historical[competitor_number][previous_non_zero_demand_time_periods[k]]:
						our_rank=our_rank+1
			
			#the calculations below COULD be speeded up by storing binomial coefficients
			#calculate the predicted demand portion given parameters a,b
			#cases_total=0
			#total_wins=0
			for r in range(int(max(1,params[0])), int(params[1]+1)):
				#print('C',C,', our_rank=',our_rank,', r=',r)
				
				#cases_total=cases_total+out_of
				if (C-our_rank)-(r-1)>=0:
					out_of=factorial(C)/(factorial(r)*factorial(C-r))
					equationNumberOfWins=(factorial((C-our_rank))/(factorial(r-1)*factorial((C-our_rank)-(r-1))));
					predicted_demands[k-t_first]=predicted_demands[k-t_first]+(1/(params[1]-params[0]+1))*(equationNumberOfWins/out_of)
					
			#sum of predicted demand portions
			sum_of_predicted_demand=sum_of_predicted_demand+predicted_demands[k-t_first]
			
		if (t_last-t_first)>0:
			predicted_portion_of_demand=sum_of_predicted_demand/(t_last-t_first)
		
			#if params[0]==0:
				#predicted_portion_of_demand
			
			#non_zero_demand_periods#
			if predicted_portion_of_demand>0:
				params[2]=(sum_of_actual_demand/(t_last-t_first))/	predicted_portion_of_demand
				#print(demand_vector)params,',',
				#print(params[3],',',sum_of_actual_demand,',',predicted_portion_of_demand,',',sum_of_predicted_demand,',',C,',',(t_last-t_first))

				#multiple predicted demands by predicted N
				#and calculate demand squared error
			for k in range(t_first, t_last+1): 
				predicted_demands[k-t_first]=predicted_demands[k-t_first]*params[2]
				obj=obj+(demand_historical[previous_non_zero_demand_time_periods[k]]-predicted_demands[k-t_first])**2
				#obj=obj+((demand_historical[previous_non_zero_demand_time_periods[k]]*prices_historical[competitor_number][previous_non_zero_demand_time_periods[k]])-(predicted_demands[k-t_first]*prices_historical[competitor_number][previous_non_zero_demand_time_periods[k]]))**2
				
				#obj=obj+((demand_historical[previous_non_zero_demand_time_periods[k]]-predicted_demands[k-t_first])**2)+((0.01*((demand_historical[previous_non_zero_demand_time_periods[k]]*prices_historical[competitor_number][previous_non_zero_demand_time_periods[k]])-(predicted_demands[k-t_first]*prices_historical[competitor_number][previous_non_zero_demand_time_periods[k]])))**2)
				
			#else:
				#obj=10000000000000000000
		else:
			obj=10000000000000000000
		
		return (obj, params[2])
	
	def factorial( value):
		result=1
		for i in range(value):
			result=result*(i+1)
		return result
	
	def sort( AA):
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
	def inverseErrorFunctionApprox( z):
		result=0
		sign=1
		if z<0:
			sign=-1
		aaa=0.147
		result=sign*Math.sqrt((Math.sqrt(Math.pow((2/(Math.pi*aaa))+(Math.log(1-Math.pow(z, 2))/2), 2)-((Math.log(1-Math.pow(z, 2)))/(aaa)))-(((2)/(Math.pi*aaa))+(((Math.log(1-Math.pow(z, 2))))/(2)))))
		return result
	
	#normal distribution (equations from wikipedia)
	def quantileFunction( prob, mu, sigma):
		x=0
		if prob<=0:
			x=mu-sigma*5
		elif prob>=1:
			x=mu+sigma*5
		else:
			x=mu+sigma*min(5, max(-5, Math.sqrt(2)*inverseErrorFunctionApprox(2*prob-1)))
		return x
		
	#normal distribution (equations from wikipedia)
	def ZFunction(prob):
		z=0
		if prob<=0:
			z=-5
		elif prob>=1:
			z=5
		else:
			z=min(5, max(-5, Math.sqrt(2)*inverseErrorFunctionApprox(2*prob-1)))
		return z
		
	#sorted prices could provide a more stable model, especially as competitor prices will not in general be modelled well with an exponential smoothing model for each individual customer 
	
		

	def next_price( predicted_comp_prices):
		#print(predicted_comp_prices)
		#equal probability interval sample
		wtp_sample_values=[]	
		for i in range(WTPSS):
			wtp_sample_values.append(max(1, min(100,quantileFunction(i/(WTPSS-1), MU_1, SIGMA_1))))
		
		next_price=np.random.uniform(1,100)
		
		#generate a set of potential prices. The build the demand vector for each of these
		#sort the forecast prices
		#+/-5 the most expensive, cheapest
		#each price and the mid points
		
		#sort the forecast competitor prices 
		[sorted_forecast_prices,ind_ord]=sort(predicted_comp_prices)
		
		potential_prices=[]
		for i in range(WTPSS):
			potential_prices.append(wtp_sample_values[i])
			
		
		potential_prices.append(max(1,min(100,sorted_forecast_prices[0]-5)))
		potential_prices.append(max(1,min(100,sorted_forecast_prices[C-2]+5)))
		potential_prices.append(max(1,min(100,sorted_forecast_prices[0]-15)))
		potential_prices.append(max(1,min(100,sorted_forecast_prices[C-2]+15)))
		for i in range(len(sorted_forecast_prices)):
			potential_prices.append(max(1,min(100,sorted_forecast_prices[i])))
			if i<len(sorted_forecast_prices)-1:
				potential_prices.append(max(1,min(100,((sorted_forecast_prices[i]+sorted_forecast_prices[i+1])/2))))
		
		#build and evaluate the demand for each of these potential prices
		best_profit=-1
		for k in range(len(potential_prices)):
			#the profit associated with charging our own predicted price
			#generate a demand vector for each price from the wtp sample
			sum=0
			demand_vector=[0 for i in range(C)]
			for i in range(C-1):#for each price
				for j in range(WTPSS):#for each sampled customer wtp
					if predicted_comp_prices[i]<wtp_sample_values[j]:#then there is a change theat this customer will purchase from this competitor
						contribution=((wtp_sample_values[j]-predicted_comp_prices[i])/wtp_sample_values[j])**B_1
						if wtp_sample_values[j]<=0:
							contribution=0
						sum=sum+contribution
						demand_vector[i]=demand_vector[i]+contribution
			
			#demand vector contribution from our price (we are index C-1)
			for j in range(WTPSS):#for each sampled customer wtp
				if potential_prices[k]<wtp_sample_values[j]:#then there is a change theat this customer will purchase from this competitor
					contribution=((wtp_sample_values[j]-potential_prices[k])/wtp_sample_values[j])**B_1
					if wtp_sample_values[j]<=0:
						contribution=0
					sum=sum+contribution
					demand_vector[C-1]=demand_vector[C-1]+contribution
			
			#expected_demand=0 calculate this from observed demands for use in correcting model parameters. Below we are maximising demand magnitude without knowing the size of the population, but using the normal wtp and A*((wtp-price)/wtp)^B model. No attempt is made to estimate the size of the population (but surely this matters)
			if sum>0:
				for i in range(C):#for each price
					demand_vector[i]=demand_vector[i]/sum
					

				prof=demand_vector[C-1]*potential_prices[k]
				if prof>best_profit:
					best_profit=prof
					next_price=potential_prices[k]
		
		return next_price	
		
	def next_price_demand_model_2( predicted_comp_prices):
		#print(predicted_comp_prices)
		
		
		##the expected profit for undercutting each competitor by a bit and also charging a higher price (being ranked last, the probability only has to be calculated once) Highest price competitor plus 5 as well
		
		#find the demand portion associated with each rank
		rank_demand=[0]*C
		for i in range(C):
			for r in range(int(max(1,A_3)), int(B_3+1)):
				if ((C-(i+1))-(r-1))>=0:
					equationNumberOfWins=(factorial((C-(i+1)))/(factorial(r-1)*factorial((C-(i+1))-(r-1))));
					
					out_of=factorial(C)/(factorial(r)*factorial(C-r))
					
					rank_demand[i]=rank_demand[i]+((1/(B_3-A_3+1))*(equationNumberOfWins/out_of))
				#if B_3-A_3>0:*N
					#rank_demand[i]=rank_demand[i]+(((1/(B_3-A_3))*(equationNumberOfWins/out_of))*N)
				#elif B_3-A_3==0:
					#rank_demand[i]=rank_demand[i]+(((equationNumberOfWins/out_of))*N)
				
		
		#get the rank order of the prices 
		[sorted_forecast_prices,ind_ord]=sort(predicted_comp_prices)
		#print(rank_demand)
		#print(sorted_forecast_prices)
		
		#the expected profit for undercutting each competitor
		max_prof=-1
		next_price=np.random.uniform(1,100)
		for i in range(len(sorted_forecast_prices)):
			comp_price=sorted_forecast_prices[i]
			prof_of_undercut=(comp_price-1)*rank_demand[i]
			if prof_of_undercut>max_prof:
				max_prof=prof_of_undercut
				next_price=comp_price-1
		
		#the profit associated with being the most expensive
		#(100)
		#prof_of_overcut=(sorted_forecast_prices[C-2]+5)*rank_demand[C-1]
		prof_of_overcut=100*rank_demand[C-1]
		if prof_of_overcut>max_prof:
			max_prof=prof_of_overcut
			next_price=100#sorted_forecast_prices[C-2]+5#100#
		
		#print(next_price)
		
		return next_price
	
	
	def next_price_demand_model_3( predicted_comp_prices):
		#print(predicted_comp_prices)
		#equal probability interval sample
		wtp_sample_values=[]	
		for i in range(WTPSS):
			wtp_sample_values.append(max(1,min(100,quantileFunction(i/(WTPSS-1), MU_4, SIGMA_4))))
		
		next_price=np.random.uniform(1,100)
		
		#generate a set of potential prices. The build the demand vector for each of these
		#sort the forecast prices
		#+/-5 the most expensive, cheapest
		#each price and the mid points
		
		#sort the forecast competitor prices 
		[sorted_forecast_prices,ind_ord]=sort(predicted_comp_prices)
		
		potential_prices=[]
		for i in range(WTPSS):
			potential_prices.append(wtp_sample_values[i])
		
		potential_prices.append(max(1,min(100,sorted_forecast_prices[0]-5)))
		potential_prices.append(max(1,min(100,sorted_forecast_prices[C-2]+5)))
		potential_prices.append(max(1,min(100,sorted_forecast_prices[0]-15)))
		potential_prices.append(max(1,min(100,sorted_forecast_prices[C-2]+15)))
		for i in range(len(sorted_forecast_prices)):
			potential_prices.append(max(1,min(100,sorted_forecast_prices[i])))
			if i<len(sorted_forecast_prices)-1:
				potential_prices.append(max(1,min(100,((sorted_forecast_prices[i]+sorted_forecast_prices[i+1])/2))))
		
		#build and evaluate the demand for each of these potential prices
		best_profit=-1
		for k in range(len(potential_prices)):
			#the profit associated with charging our own predicted price
			#generate a demand vector for each price from the wtp sample
			sum=0
			demand_vector=[0 for i in range(C)]
			for i in range(C-1):#for each price
				for j in range(WTPSS):#for each sampled customer wtp
					if predicted_comp_prices[i]<wtp_sample_values[j]:#then there is a change theat this customer will purchase from this competitor
						contribution=max(0,(1-((wtp_sample_values[j]-predicted_comp_prices[i])/wtp_sample_values[j])))**B_4
						if wtp_sample_values[j]<=0:
							contribution=0
						sum=sum+contribution
						demand_vector[i]=demand_vector[i]+contribution
			
			#demand vector contribution from our price (we are index C-1)
			for j in range(WTPSS):#for each sampled customer wtp
				if potential_prices[k]<wtp_sample_values[j]:#then there is a change theat this customer will purchase from this competitor
					contribution=(1-((wtp_sample_values[j]-potential_prices[k])/wtp_sample_values[j]))**B_4
					if wtp_sample_values[j]<=0:
						contribution=0
						print('that was invalid')
					sum=sum+contribution
					demand_vector[C-1]=demand_vector[C-1]+contribution
			
			#expected_demand=0 calculate this from observed demands for use in correcting model parameters. Below we are maximising demand magnitude without knowing the size of the population, but using the normal wtp and A*((wtp-price)/wtp)^B model. No attempt is made to estimate the size of the population (but surely this matters)
			if sum>0:
				for i in range(C):#for each price
					demand_vector[i]=demand_vector[i]/sum
					

				prof=demand_vector[C-1]*potential_prices[k]
				if prof>best_profit:
					best_profit=prof
					next_price=potential_prices[k]
		
		return next_price	
	
	def update_exp_smooth_params_return_forecast_prices( comp_prices_last_t, t):
		for c in range(C-1):
			Base_value[t-1][c]=(alpha*comp_prices_last_t[c])+((1-alpha)*(Base_value[t-2][c]+Trend[t-2][c]))
			
			Trend[t-1][c]=(beta*(Base_value[t-1][c]-Base_value[t-2][c]))+((1-beta)*Trend[t-2][c])
			
			prices_next_t[c]=max(1, min(100,Base_value[t-2][c]+Trend[t-1][c]))
		return prices_next_t
		
	
	################################
	np.random.seed(int(time.time()))
	
	#work out the time period from the size of the historic demand vector
	t=len(demand_historical)
	
	# if it's the first day
	#if demand_historical.size == 0:
	if t == 0:
		#print(len(parameterdump))
		
		#we are the first row of prices_historical (competitor number/row=0)
		competitor_number=0
		
		#print(competitor_number)
		#this is a way to use this to model each demand model alone
		#for testing purposes
		fix_arm=parameterdump[1]
		demand_model_to_use=parameterdump[2]
		
		#forecast model smoothing parameters
		alpha=0.2#parameterdump[3]#
		beta=0.2#parameterdump[4]#
		
		epsilon_1=0.2 #parameterdump[5]#probability that a random price is chosen
		epsilon_2=0.4 #parameterdump[6]#probability that a random price is chosen
		
		#number of competitors
		C=len(prices_historical)
		
		
			
		
		############################
		Base_value=[[0 for j in range(C-1)] for i in range(1000)]
		Trend=[[0 for j in range(C-1)] for i in range(1000)]
		prices_next_t=[0 for i in range(C-1)]
		other_prices=[0 for i in range(C-1)]
		#demand model 1
		#initial parameter estimates
		B_1=2
		SIGMA_1=5
		MU_1=25
		
		#cheapest of subset demand model
		#set a<=b<=C parameter bounds
		B_3=max(1,round(C))
		A_3=round(B_3/2)
		
		
		#demand model 1
		MU_4=25
		SIGMA_4=5
		B_4=2
		
		

		WTPSS=20
		
		ZScores=[0]*WTPSS
		for i in range(WTPSS):
			prob=i/(WTPSS-1)
			ZScores=ZFunction(prob)
		

		#remember to set parameter settings based on the number of competitors
		
		#a second implementation of simulated annealing for the demand model parameters to previous observations  
		#mu=[0,100], signma=[0, 20], B=[0,5], N=[free parameter, multiplicative step length] (for a given model the N that fits best can be directly calculated)
		parameters_2=3
		iterations_2=2
		initial_step_lengths_2=[]#t==0   f(number of iterations, number of time periods)
		#parameter_selection_distribution_2=[0.33, 0.66, 1]
		parameter_selection_distribution_2=[0.33,0.66, 1]
		parameter_selection_distribution_2_step_len=[0.33,0.33, 0.33]
		param_bounds_2=[[1,100],[1, 30],[0.1,5]]
		t0Factor_2=0.001
		max_param_ranges_2=[99, 29, 4.9]
		times_across_space_2=5;#using a linearly decreasing time step
		
		parameters_3=2
		iterations_3=2
		initial_step_lengths_3=[]
		#parameter_selection_distribution_2=[0.33, 0.66, 1]
		parameter_selection_distribution_3=[0.5,1]
		parameter_selection_distribution_3_step_len=[0.5,0.5]
		param_bounds_3=[[0,100],[0,100]]
		t0Factor_3=0.001
		max_param_ranges_3=[100, 100]
		times_across_space_3=5;#using a linearly decreasing time step
		
		parameters_4=3
		iterations_4=2
		initial_step_lengths_4=[]#t==0   f(number of iterations, number of time periods)
		#parameter_selection_distribution_2=[0.33, 0.66, 1]
		parameter_selection_distribution_4=[0.33,0.66, 1]
		parameter_selection_distribution_4_step_len=[0.33,0.33, 0.33]
		param_bounds_4=[[1,100],[1, 30],[0.1,5]]
		t0Factor_4=0.001
		max_param_ranges_4=[99, 29, 4.9]
		times_across_space_4=5;#using a linearly decreasing time step
		
		#wtp demand model
		unitarySumOfDecreasingStepLengths=0
		for i in range(iterations_2*1000):
			unitarySumOfDecreasingStepLengths=unitarySumOfDecreasingStepLengths+(1-(i/(iterations_2*1000)))
		
		initial_step_lengths_2=np.zeros((parameters_2))
		for i in range(parameters_2):
			initial_step_lengths_2[i]=(times_across_space_2*max_param_ranges_2[i])/(parameter_selection_distribution_2_step_len[i]*unitarySumOfDecreasingStepLengths)
			#initial_step_lengths_2[i]=0.01;
		
		#cheapest of subset demand model	
		unitarySumOfDecreasingStepLengths=0
		for i in range(iterations_3*1000):
			unitarySumOfDecreasingStepLengths=unitarySumOfDecreasingStepLengths+(1-(i/(iterations_3*1000)))
		
		initial_step_lengths_3=np.zeros((parameters_3))
		for i in range(parameters_3):
			initial_step_lengths_3[i]=(times_across_space_3*max_param_ranges_3[i])/(parameter_selection_distribution_3_step_len[i]*unitarySumOfDecreasingStepLengths)
			#initial_step_lengths_2[i]=0.01;
			
		#wtp2 demand model
		unitarySumOfDecreasingStepLengths=0
		for i in range(iterations_4*1000):
			unitarySumOfDecreasingStepLengths=unitarySumOfDecreasingStepLengths+(1-(i/(iterations_4*1000)))
		
		initial_step_lengths_4=np.zeros((parameters_4))
		for i in range(parameters_4):
			initial_step_lengths_4[i]=(times_across_space_2*max_param_ranges_4[i])/(parameter_selection_distribution_4_step_len[i]*unitarySumOfDecreasingStepLengths)
			#initial_step_lengths_2[i]=0.01;
		
		
		
		#previous time periods taken into account in parameter fitting
		PPU=100
		
		
		counts_1 = [0 for col in range(4)] # a vector with 10 entries showing how many times each price is chosen
		values_1 = [0.0 for col in range(4)] # a vector with 10 entries showing the average reward(demand) obtained by each price
		index_last_period_1=0
		
		
		
		counts_2 = [0 for col in range(10)] # a vector with 10 entries showing how many times each price is chosen
		values_2 = [0.0 for col in range(10)] # a vector with 10 entries showing the average reward(demand) obtained by each price
		index_last_period_2=0
		#the code is for epsilon-greedy
		#in epsilon-decreasing epsilon=min{1,epsilon0/t}

		#prices_2=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100] #we test these 10 prices to compare their demand 
		prices_2=[5,15,25,35,45,55,65,75,85,95]
		############################
		

		popt = np.random.uniform(1,100)
		
		#MAB_2
		index_2=max(0, min(len(prices_2)-1, math.floor(popt/10)))
		#counts_2[index_2]=counts_2[index_2]+1 
		index_last_period_2=index_2
		
		
		previous_non_zero_demand_time_periods=[]
		
		
		#set the structure of parameter dump
		parameterdump=[competitor_number, fix_arm, demand_model_to_use, alpha, beta, epsilon_1, epsilon_2, C, Base_value, Trend, prices_next_t, other_prices, B_1, SIGMA_1, MU_1, B_3, A_3, MU_4, SIGMA_4, B_4, WTPSS, ZScores, parameters_2, iterations_2, initial_step_lengths_2, parameter_selection_distribution_2, parameter_selection_distribution_2_step_len, param_bounds_2, t0Factor_2, max_param_ranges_2, times_across_space_2, parameters_3, iterations_3, initial_step_lengths_3, parameter_selection_distribution_3, parameter_selection_distribution_3_step_len, param_bounds_3, t0Factor_3, max_param_ranges_3, times_across_space_3, parameters_4, iterations_4, initial_step_lengths_4, parameter_selection_distribution_4, parameter_selection_distribution_4_step_len, param_bounds_4, t0Factor_4, max_param_ranges_4, times_across_space_4, PPU, counts_1, values_1, index_last_period_1, counts_2, values_2, index_last_period_2, prices_2, previous_non_zero_demand_time_periods]

		
	elif t<2:
		#unpack parameterdump
		[competitor_number, fix_arm, demand_model_to_use, alpha, beta, epsilon_1, epsilon_2, C, Base_value, Trend, prices_next_t, other_prices, B_1, SIGMA_1, MU_1, B_3, A_3, MU_4, SIGMA_4, B_4, WTPSS, ZScores, parameters_2, iterations_2, initial_step_lengths_2, parameter_selection_distribution_2, parameter_selection_distribution_2_step_len, param_bounds_2, t0Factor_2, max_param_ranges_2, times_across_space_2, parameters_3, iterations_3, initial_step_lengths_3, parameter_selection_distribution_3, parameter_selection_distribution_3_step_len, param_bounds_3, t0Factor_3, max_param_ranges_3, times_across_space_3, parameters_4, iterations_4, initial_step_lengths_4, parameter_selection_distribution_4, parameter_selection_distribution_4_step_len, param_bounds_4, t0Factor_4, max_param_ranges_4, times_across_space_4, PPU, counts_1, values_1, index_last_period_1, counts_2, values_2, index_last_period_2, prices_2, previous_non_zero_demand_time_periods]=parameterdump

		#store demand time period index if there was non-zero demand
		#
		if demand_historical[t-1]>0:
			previous_non_zero_demand_time_periods.append(t-1)
		
		#initialise base value (trend initialised to 0)
		#for c in range(C):
		counter=0
		for i in range(C):
			if i!=competitor_number:
				other_prices[counter]=prices_historical[i,t-1]
				counter=counter+1
		
		[sorted_prices, ind_order]=sort(other_prices)#prices_historical[:,t-1]
		Base_value[0]=sorted_prices
		
		
		
		##MAB_2 update
		
		
		#random initial price
		popt = np.random.uniform(1,100)
		
		
		#update MAB_2
		values_2[int(index_last_period_2)]=update_2(int(index_last_period_2),demand_historical[t-1]*prices_historical[competitor_number,t-1])
		#
		index_2=max(0, min(len(prices_2)-1, math.floor(popt/10)))
		#counts_2[index_2]=counts_2[index_2]+1 #the number of times this price is chosen increases by 1
		index_last_period_2=index_2
		
		
		#set parameterdump ready for the next time period
		parameterdump=[competitor_number, fix_arm, demand_model_to_use, alpha, beta, epsilon_1, epsilon_2, C, Base_value, Trend, prices_next_t, other_prices, B_1, SIGMA_1, MU_1, B_3, A_3, MU_4, SIGMA_4, B_4, WTPSS, ZScores, parameters_2, iterations_2, initial_step_lengths_2, parameter_selection_distribution_2, parameter_selection_distribution_2_step_len, param_bounds_2, t0Factor_2, max_param_ranges_2, times_across_space_2, parameters_3, iterations_3, initial_step_lengths_3, parameter_selection_distribution_3, parameter_selection_distribution_3_step_len, param_bounds_3, t0Factor_3, max_param_ranges_3, times_across_space_3, parameters_4, iterations_4, initial_step_lengths_4, parameter_selection_distribution_4, parameter_selection_distribution_4_step_len, param_bounds_4, t0Factor_4, max_param_ranges_4, times_across_space_4, PPU, counts_1, values_1, index_last_period_1, counts_2, values_2, index_last_period_2, prices_2, previous_non_zero_demand_time_periods]
		
	else:
		#unpack parameterdump
		[competitor_number, fix_arm, demand_model_to_use, alpha, beta, epsilon_1, epsilon_2, C, Base_value, Trend, prices_next_t, other_prices, B_1, SIGMA_1, MU_1, B_3, A_3, MU_4, SIGMA_4, B_4, WTPSS, ZScores, parameters_2, iterations_2, initial_step_lengths_2, parameter_selection_distribution_2, parameter_selection_distribution_2_step_len, param_bounds_2, t0Factor_2, max_param_ranges_2, times_across_space_2, parameters_3, iterations_3, initial_step_lengths_3, parameter_selection_distribution_3, parameter_selection_distribution_3_step_len, param_bounds_3, t0Factor_3, max_param_ranges_3, times_across_space_3, parameters_4, iterations_4, initial_step_lengths_4, parameter_selection_distribution_4, parameter_selection_distribution_4_step_len, param_bounds_4, t0Factor_4, max_param_ranges_4, times_across_space_4, PPU, counts_1, values_1, index_last_period_1, counts_2, values_2, index_last_period_2, prices_2, previous_non_zero_demand_time_periods]=parameterdump
		#update demand model parameters (correction based on observations)
			
		if demand_historical[t-1]>0:
			previous_non_zero_demand_time_periods.append(t-1)
			
		#print('prices_historical[competitor_number,t-1]=',prices_historical[competitor_number,t-1])
		if t>2:
			if t==50:
				epsilon_2=0.2
				counts_2 = [0 for col in range(10)] # a vector with 10 entries showing how many times each price is chosen
				values_2 = [0.0 for col in range(10)] # a vector with 10 entries showing the average reward(demand) obtained by each price
			
			if t>50:
				values_1[index_last_period_1]=update(index_last_period_1,demand_historical[t-1]*prices_historical[competitor_number,t-1])
				index=select_arm(epsilon_1,values_1) #index is the price chosen at t###(epsilon_1*(1-(t/1000)))
			else:
				index=3
		elif t==2:
			index=3
		
		
		
		#DEMAND MODEL 1
		######################################################
		#THE CURRENT DEMAND PROFILE FITTING MODEL IS A SEVERE BOTTLE NECK
		#MAYBE MULTI-ARMED BANDITS CAN BE USED TO SEARCH THE DEMAND MODEL SPACE
		
		if t>25:
		
			initial_parameters_2=[MU_1, SIGMA_1, B_1, 1]
			
			#current ones are used as the initial solution
			updated_params_2=simulated_annealing_all_demand_models(1, iterations_2, parameters_2, initial_parameters_2, initial_step_lengths_2, parameter_selection_distribution_2, param_bounds_2, t0Factor_2, prices_historical, demand_historical, previous_non_zero_demand_time_periods, t, np)
			
			[MU_1, SIGMA_1, B_1, predicted_n_1]=updated_params_2
			#####################################################
				
			#elif demand_model_to_use==2:
			
			#DEMAND MODEL 2
			######################################################
			#MAYBE MULTI-ARMED BANDITS CAN BE USED TO SEARCH THE DEMAND MODEL SPACE
			initial_parameters_3=[A_3, B_3, 1]
			
			#current ones are used as the initial solution
			updated_params_3=simulated_annealing_all_demand_models(2, iterations_3, parameters_3, initial_parameters_3, initial_step_lengths_3, parameter_selection_distribution_3, param_bounds_3, t0Factor_3, prices_historical, demand_historical, previous_non_zero_demand_time_periods, t, np)
			
			[A_3, B_3, predicted_n_1]=updated_params_3
			#####################################################
			
			
			#DEMAND MODEL 3
			######################################################
			#THE CURRENT DEMAND PROFILE FITTING MODEL IS A SEVERE BOTTLE NECK
			#MAYBE MULTI-ARMED BANDITS CAN BE USED TO SEARCH THE DEMAND MODEL SPACE
			initial_parameters_4=[MU_4, SIGMA_4, B_4, 1]
			
			#current ones are used as the initial solution
			updated_params_4=simulated_annealing_all_demand_models(3, iterations_4, parameters_4, initial_parameters_4, initial_step_lengths_4, parameter_selection_distribution_4, param_bounds_4, t0Factor_4, prices_historical, demand_historical, previous_non_zero_demand_time_periods, t, np)
			
			[MU_4, SIGMA_4, B_4, predicted_n_1]=updated_params_4
			#####################################################
		
		
		
		
		
		#demand model epsilon greedy is one to try
		#MAB_2 update reward vector
		#print('index=',index_last_period_2)
		values_2[int(index_last_period_2)]=update_2(int(index_last_period_2),demand_historical[t-1]*prices_historical[competitor_number,t-1])
		
		

		#prices offered in the previous time period
		counter=0
		for i in range(C):
			if i!=competitor_number:
				other_prices[counter]=prices_historical[i,t-1]
				counter=counter+1
		
		[sorted_prices, ind_order]=sort(other_prices)#prices_historical[:,t-1]
		#demand_model_to_use=2
		
		forecast_price_set=update_exp_smooth_params_return_forecast_prices(sorted_prices, t)
		#sample wtp and derive competitor probability distribution
		#find optimum d*p
		#demand_model_to_use=1
		#print(ind_order)
		if demand_model_to_use==1:
			popt = next_price(forecast_price_set)
			index_2=max(0, min(len(prices_2)-1, math.floor(popt/10)))
		elif demand_model_to_use==2:
			popt = next_price_demand_model_2(forecast_price_set)
			index_2=max(0, min(len(prices_2)-1, math.floor(popt/10)))
		elif demand_model_to_use==3:	
			popt = next_price_demand_model_3(forecast_price_set)
			index_2=max(0, min(len(prices_2)-1, math.floor(popt/10)))
		elif demand_model_to_use==4:
			index_2=select_arm(epsilon_2,values_2) #index is the price chosen at t
			popt = prices_2[index_2]
			
		#counts_2[index_2]=counts_2[index_2]+1 #the number of times this price is chosen increases by 1
		#update last index ready for next time period
		index_last_period_2=index_2	
		#print('popt=',popt)
		
		#set parameterdump ready for the next time period
		parameterdump=[competitor_number, fix_arm, demand_model_to_use, alpha, beta, epsilon_1, epsilon_2, C, Base_value, Trend, prices_next_t, other_prices, B_1, SIGMA_1, MU_1, B_3, A_3, MU_4, SIGMA_4, B_4, WTPSS, ZScores, parameters_2, iterations_2, initial_step_lengths_2, parameter_selection_distribution_2, parameter_selection_distribution_2_step_len, param_bounds_2, t0Factor_2, max_param_ranges_2, times_across_space_2, parameters_3, iterations_3, initial_step_lengths_3, parameter_selection_distribution_3, parameter_selection_distribution_3_step_len, param_bounds_3, t0Factor_3, max_param_ranges_3, times_across_space_3, parameters_4, iterations_4, initial_step_lengths_4, parameter_selection_distribution_4, parameter_selection_distribution_4_step_len, param_bounds_4, t0Factor_4, max_param_ranges_4, times_across_space_4, PPU, counts_1, values_1, index_last_period_1, counts_2, values_2, index_last_period_2, prices_2, previous_non_zero_demand_time_periods]
		
	return (popt, parameterdump)