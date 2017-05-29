#!/usr/bin/env python
import random #see https://docs.python.org/2/library/random.html for lots of the feature of random
import numpy as np
import matplotlib.pyplot as plt

from Fixed_price_competitor import *
from Random_price_competitor import *
from Epsilon_greedy_competitor import *
from Epsilon_greedy_competitor2 import *
from demand_model_1 import *
from demand_model_2 import *
from demand_model_3 import *
from demand_model_5 import *
from Mode_price_forecast_competitor import *
from Sine_competitor import *
from Four_armed_bandit_2 import *
from Four_armed_bandit_3 import *


#x=np.array([[7,8,5],[3,5,7]],np.int32)
#plt.plot(x[:,0],x[:,1])
#plt.show()

repeats=1;

C=0;#number of competitors, initialised below

T=1000#time periods
N=500#customer population size
arrival_rate=0.5#average arrival rate in each time period from the population of customers 



np.random.seed(0)

#list of object derived from the Competitor class
competitor_objs=[]

#which competitors to include
use_random=True#False#
use_fixed=True#False#
use_epsilon_greedy_1=True#False#
use_demand_profile_cheapest=True#False#
use_mode_price=True#False#
use_sine_wave=True#False#
use_four_armed_bandit=True#False#
use_epsilon_greedy_2=True#False#
use_demand_profile_bargain_wtp=True#False#
use_demand_profile_quality_wtp=True#False#

exp_name='various_demand_models_duopoly_2'
fileName='./results/'+"total_profit_"+exp_name+".txt"
f=open(fileName,'w')
#alpha
#parameters_to_test=[0.1,0.2,0.3,0.4,0.5]
parameters_to_test=[[50,15]]
#parameters_to_test_2=[[20,5],[20,10],[20,25],[50,5],[50,10],[50,25],[80,5],[80,10],[80,25]]

for tp in range(len(parameters_to_test)):
	#print('parameters_to_test=',parameters_to_test[tp])
	print('parameters_to_test, mu=',parameters_to_test[tp][0],', signma=',parameters_to_test[tp][1])
	f.write('parameters_to_test, mu=,'+str(parameters_to_test[tp][0])+', signma='+str(parameters_to_test[tp][1])+"\n")#", "+
	for l in range(3):#3
		print('demand model ',(l+1))
		f.write('demand model '+str((l+1))+"\n")#", "+
		
		comp_index_count=0
		competitor_objs=[]
		competitor_names=[]
		parameterdumps=[]

		#Random competitor
		if use_random:
			rand_comp_1=Random_price_competitor()#comp_index_count
			competitor_objs.append(rand_comp_1)
			parameterdumps.append([comp_index_count])
			competitor_names.append('Random')
			comp_index_count=comp_index_count+1

		#Fixed price competitor
		if use_fixed:
			fixed_comp_1=Fixed_price_competitor()#comp_index_count, 50
			competitor_objs.append(fixed_comp_1)
			parameterdumps.append([comp_index_count])
			competitor_names.append('Fixed')
			comp_index_count=comp_index_count+1

		#epsilon competitor
		epsilon=0.1
		if use_epsilon_greedy_1:
			epsilon_greedy_comp_1=Epsilon_greedy_competitor()#comp_index_count, epsilon
			competitor_objs.append(epsilon_greedy_comp_1)
			parameterdumps.append([comp_index_count])
			competitor_names.append('Epsilon_greedy')
			comp_index_count=comp_index_count+1

		#demand profile competitor (depreciated)
		#price_profile_comp_1=demand_profile_competitor(comp_index_count, np)
		#competitor_objs.append(price_profile_comp_1)
		#competitor_names.append('price_profile_1')
		#comp_index_count=comp_index_count+1

		#demand profile model (own prices removed from exponential smoothing)
		if use_demand_profile_cheapest:
			price_profile_comp_3=Four_armed_bandit_3()#comp_index_count,0.5,0.2,np
			competitor_objs.append(price_profile_comp_3)
			parameterdumps.append([comp_index_count, True, 2])
			competitor_names.append('price_profile_cheapest_subset')
			comp_index_count=comp_index_count+1

		#mode price forecast competitor
		if use_mode_price:
			mode_price_forecast_comp=Mode_price_forecast_competitor()#comp_index_count
			competitor_objs.append(mode_price_forecast_comp)
			parameterdumps.append([comp_index_count])
			competitor_names.append('mode_price')
			comp_index_count=comp_index_count+1

		#sine wave competitor
		if use_sine_wave:
			sine_wave_comp=Sine_competitor()#comp_index_count
			competitor_objs.append(sine_wave_comp)#
			parameterdumps.append([comp_index_count])
			competitor_names.append('sine_wave')
			comp_index_count=comp_index_count+1
			
		if use_four_armed_bandit:
			demand_model_bandit_comp_3=Four_armed_bandit_3()#comp_index_count,0.5,0.2,np
			competitor_objs.append(demand_model_bandit_comp_3)
			parameterdumps.append([comp_index_count, False,-1])
			competitor_names.append('four_armed_bandit')
			comp_index_count=comp_index_count+1
			
		if use_demand_profile_bargain_wtp:
			#demand_model_1_comp=Four_armed_bandit_2()#comp_index_count,0.5,0.2,np
			demand_model_1_comp=Four_armed_bandit_3()#comp_index_count,0.5,0.2,np
			competitor_objs.append(demand_model_1_comp)
			parameterdumps.append([comp_index_count, True, 1])
			competitor_names.append('price_profile_WTP bargain')
			comp_index_count=comp_index_count+1
			
		if use_demand_profile_quality_wtp:
			demand_model_3_comp=Four_armed_bandit_3()#comp_index_count,0.5,0.2,np
			competitor_objs.append(demand_model_3_comp)
			parameterdumps.append([comp_index_count, True, 3])
			competitor_names.append('price_profile_WTP quality')
			comp_index_count=comp_index_count+1

		#use epsilon greedy 2/library/random
		if use_epsilon_greedy_2:
			epsilon_greedy_comp_2=Epsilon_greedy_competitor2()#comp_index_count,epsilon
			competitor_objs.append(epsilon_greedy_comp_2)
			parameterdumps.append([comp_index_count])
			competitor_names.append('epsilon_greedy_2')
			comp_index_count=comp_index_count+1

		C=len(competitor_objs);#number of competitors



		#DEMAND MODEL INITIALISATION
		#model parameters
		a=1;
		b=2;
		#normal willingness to pay distribution
		#normal willingness to pay distribution
		mu=parameters_to_test[tp][0]#75
		sigma=parameters_to_test[tp][1]
		
		#demand model 1
		#if l==0:
			#dm_1=demand_model_1(2, a, b, mu, sigma)
		#else:
			#dm_1=demand_model_5(2, a, b, mu, sigma)
			
		#demand model 1
		if l==0:
			dm_1=demand_model_1(2, a, b, mu, sigma)
		elif l==1:
			dm_1=demand_model_3(2, 0, 2)#parameterised version of the above, a=min subset size, b=max subset size
		else:
			dm_1=demand_model_5(2, a, b, mu, sigma)
		
		#non-dynamic randomly generated customer prices
		prices_historical=np.zeros((C,T))

		#print(prices_historical)

		#demand per competitor in each time period
		comp_demand=np.zeros((C,T))
		#total profit per competitor
		comp_profit=np.zeros((C,T))
		#total profit per competitor (matrix)
		comp_profit_matrix=np.zeros((C,C,T))

		#initialise array for the customer competitor cumulative distribution
		cust_comp_select_cumu_dist=np.zeros((C))
		sum_cumu_dist=0
		selected_comp_index=0

		me_ind=C-4
		
		#repeat runs
		for rep in range(repeats):
			print('rep=',rep)
			for c1 in range(me_ind,me_ind+1):#,C-1
				for c2 in range(C):#c1+1,
					if c1!=c2:
						#reset demand array
						comp_demand_p_input=np.zeros((C,T))
						#reset prices_historical
						prices_historical=np.zeros((C,T))
						#the two competitors
						competitor_set=[c1,c2]
						#time steps
						print(competitor_names[c1],' vs ',competitor_names[c2])
						for t in range(T):
							#print(t)
							#get competitor prices for the current time period (current/next: check whic for actual competition. In this version competit)
							prices_this_t=[]#prices this time period (to avoid contaminating the timeline)
							for c_number in competitor_set:
								(ppt, parameterdumps[c_number])=competitor_objs[c_number].p(prices_historical, comp_demand_p_input[c_number], parameterdumps[c_number], t)
								prices_this_t.append(ppt)
							#customer arrival process
							for k in range(N):
								#customer chooses a competitor or refuses to buy.
								#winning_competitor=f(competitor prices)
								#-1 if no purchase made
								rand_arrival_number=np.random.uniform(0,1,1)
								if rand_arrival_number<arrival_rate:
									#print('hello',rand_arrival_number)
									selected_comp_index=dm_1.winning_competitor(prices_this_t, np)
									if selected_comp_index>-1:
										#update the demand and profit of the competitor who won the customer's business
										comp_demand_p_input[competitor_set[selected_comp_index]][t]=comp_demand_p_input[competitor_set[selected_comp_index]][t]+1
										comp_demand[competitor_set[selected_comp_index]][t]=comp_demand[competitor_set[selected_comp_index]][t]+1
										comp_profit[competitor_set[selected_comp_index]][t]=comp_profit[competitor_set[selected_comp_index]][t]+prices_this_t[selected_comp_index]
										comp_profit_matrix[competitor_set[selected_comp_index]][competitor_set[selected_comp_index-1]][t]=comp_profit_matrix[competitor_set[selected_comp_index]][competitor_set[selected_comp_index-1]][t]+prices_this_t[selected_comp_index]
										comp_profit_matrix[competitor_set[selected_comp_index]][competitor_set[selected_comp_index-1]][t]=comp_profit_matrix[competitor_set[selected_comp_index]][competitor_set[selected_comp_index-1]][t]+prices_this_t[selected_comp_index]
									#else:
										#all competitors are too highly priced for this customer
							#record prices on offer in the current time period
							for c in range(2):
								prices_historical[competitor_set[c]][t]=prices_this_t[c]
		total_profit_me=[0 for i in range(C)]
		total_profit_comp=[0 for i in range(C)]
		for i in range(C):
			for t in range(T):
				total_profit_me[i]=total_profit_me[i]+comp_profit_matrix[me_ind][i][t]
				total_profit_comp[i]=total_profit_comp[i]+comp_profit_matrix[i][me_ind][t]
		for i in range(C):
			f.write(str(total_profit_me[i])+', ')#+"\n"
		f.write("\n")#
		for i in range(C):
			f.write(str(total_profit_comp[i])+', ')#+"\n"
		f.write("\n")#
		print('duopoly results')		
		print('total_profit_me=')		
		print(total_profit_me)
		print('total_profit_comp=')
		print(total_profit_comp)
		f.write("\n")#
f.close()

