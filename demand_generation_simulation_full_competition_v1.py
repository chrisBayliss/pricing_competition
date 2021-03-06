#!/usr/bin/env python
import random #see https://docs.python.org/2/library/random.html for lots of the feature of random
import numpy as np
import matplotlib.pyplot as plt

from Fixed_price_competitor import *
from Random_price_competitor import *
from Epsilon_greedy_competitor import *
from Epsilon_greedy_competitor2 import *
from demand_profile_competitor import *
from demand_profile_competitor_exp_smooth import *
from demand_profile_competitor_cheapest_DM_exp_smooth import *
from demand_model_1 import *
from demand_model_2 import *
from demand_model_3 import *
from demand_model_4 import *
from demand_model_5 import *
from Mode_price_forecast_competitor import *
from Sine_competitor import *
from Two_armed_bandit import *
from Three_armed_bandit import *
from Four_armed_bandit_2 import *


#x=np.array([[7,8,5],[3,5,7]],np.int32)
#plt.plot(x[:,0],x[:,1])
#plt.show()

repeats=2;

C=2;#number of competitors

T=1000#time periods
N=500#customer population size
arrival_rate=0.5#average arrival rate in each time period from the population of customers 



np.random.seed(0)

#list of object derived from the Competitor class
competitor_objs=[]
#generate competitor objects (the content of parameterdump are stored within these objects in this framework. For the competition version these will be transferred to the parameterdump object (which is not used within this testing framework))

#which competitors to include
use_random=True#False#
use_fixed=True#False#
use_epsilon_greedy_1=True#False#
use_demand_profile_WTP=False#True#
use_demand_profile_cheapest=True#False#
use_mode_price=True#False#
use_sine_wave=True#False#
use_two_armed_bandit=False#True#
use_three_armed_bandit=False#True#
use_four_armed_bandit=True#False#
use_epsilon_greedy_2=True#False#




comp_index_count=0

competitor_names=[]

#Random competitor
if use_random:
	rand_comp_1=Random_price_competitor(comp_index_count)
	competitor_objs.append(rand_comp_1)
	competitor_names.append('Random')
	comp_index_count=comp_index_count+1

#Fixed price competitor
if use_fixed:
	fixed_comp_1=Fixed_price_competitor(comp_index_count, 50)
	competitor_objs.append(fixed_comp_1)
	competitor_names.append('Fixed')
	comp_index_count=comp_index_count+1

#epsilon competitor
epsilon=0.1
if use_epsilon_greedy_1:
	epsilon_greedy_comp_1=Epsilon_greedy_competitor(comp_index_count, epsilon)
	competitor_objs.append(epsilon_greedy_comp_1)
	competitor_names.append('Epsilon_greedy')
	comp_index_count=comp_index_count+1

#demand profile competitor (depreciated)
#price_profile_comp_1=demand_profile_competitor(comp_index_count, np)
#competitor_objs.append(price_profile_comp_1)
#competitor_names.append('price_profile_1')
#comp_index_count=comp_index_count+1

#demand profile competitor exponential price profile prediction (with trend)
if use_demand_profile_WTP:
	price_profile_comp_2=demand_profile_competitor_exp_smooth(comp_index_count, np)
	competitor_objs.append(price_profile_comp_2)
	competitor_names.append('price_profile_WTP')
	comp_index_count=comp_index_count+1

#demand profile model (own prices removed from exponential smoothing)
if use_demand_profile_cheapest:
	price_profile_comp_3=demand_profile_competitor_cheapest_DM_exp_smooth(comp_index_count, np)
	competitor_objs.append(price_profile_comp_3)
	competitor_names.append('price_profile_cheapest_subset')
	comp_index_count=comp_index_count+1

#mode price forecast competitor
if use_mode_price:
	mode_price_forecast_comp=Mode_price_forecast_competitor(comp_index_count)
	competitor_objs.append(mode_price_forecast_comp)
	competitor_names.append('mode_price')
	comp_index_count=comp_index_count+1

#sine wave competitor
if use_sine_wave:
	sine_wave_comp=Sine_competitor(comp_index_count)
	competitor_objs.append(sine_wave_comp)#
	competitor_names.append('sine_wave')
	comp_index_count=comp_index_count+1

#two armed badit: MAB applied to two demand models whose parameters are constantly be fit to the data
#This should mean that this approach should work nearly as well as each demand model used in its own environment
if use_two_armed_bandit:
	demand_model_bandit_comp=Two_armed_bandit(comp_index_count,0.2,np)
	competitor_objs.append(demand_model_bandit_comp)
	competitor_names.append('two_armed_bandit')
	comp_index_count=comp_index_count+1

#three armed badit: MAB applied to two demand models whose parameters are constantly be fit to the data
#This should mean that this approach should work nearly as well as each demand model used in its own environment
if use_three_armed_bandit:
	demand_model_bandit_comp_2=Three_armed_bandit(comp_index_count,0.2,0.2,np)
	competitor_objs.append(demand_model_bandit_comp_2)
	competitor_names.append('three_armed_bandit')
	comp_index_count=comp_index_count+1
	
if use_four_armed_bandit:
	demand_model_bandit_comp_3=Four_armed_bandit_2(comp_index_count,0.5,0.2,np)
	competitor_objs.append(demand_model_bandit_comp_3)
	competitor_names.append('four_armed_bandit')
	comp_index_count=comp_index_count+1

#use epsilon greedy 2/library/random
if use_epsilon_greedy_2:
	epsilon_greedy_comp_2=Epsilon_greedy_competitor2(comp_index_count,epsilon)
	competitor_objs.append(epsilon_greedy_comp_2)
	competitor_names.append('epsilon_greedy_2')
	comp_index_count=comp_index_count+1

C=len(competitor_objs);#number of competitors

#DEMAND MODEL INITIALISATION
#model parameters
a=1;
b=3;
#normal willingness to pay distribution
mu=20
sigma=5
#demand model 1
dm_1=demand_model_1(C, a, b, mu, sigma)
#dm_1=demand_model_2(C)#cheapset in uniform random subset sizes (for every arriving customer)
#dm_1=demand_model_3(C, 2, C-2)#parameterised version of the above, a=min subset size, b=max subset size
#dm_1=demand_model_4(C, mu, sigma,0)
#dm_1=demand_model_5(C, a, b, mu, sigma)

#non-dynamic randomly generated customer prices
prices_historical=np.zeros((C,T))

#print(prices_historical)

#demand per competitor in each time period
comp_demand_p_input=np.zeros((C,T))
comp_demand=np.zeros((C,T))
#total profit per competitor
comp_profit=np.zeros((C,T))

#initialise array for the customer competitor cumulative distribution
cust_comp_select_cumu_dist=np.zeros((C))
sum_cumu_dist=0
selected_comp_index=0

#repeat runs
for rep in range(repeats):
	print('rep=',rep)
	#reset demand array
	comp_demand_p_input=np.zeros((C,T))
	#reset prices_historical
	prices_historical=np.zeros((C,T))
	#time steps
	for t in range(T):
		print(t)
		#get competitor prices for the current time period (current/next: check whic for actual competition. In this version competit)
		prices_this_t=[]#prices this time period (to avoid contaminating the timeline)
		for c in range(C):
			prices_this_t.append(competitor_objs[c].p(prices_historical, comp_demand_p_input[c], t))
		#customer arrival process
		for k in range(N):
			rand_arrival_number=np.random.uniform(0,1,1)
			if rand_arrival_number<arrival_rate:
				#print('hello',rand_arrival_number)
				selected_comp_index=dm_1.winning_competitor(prices_this_t, np)
				#
				if selected_comp_index>-1:
					#update the demand and profit of the competitor who won the customer's business
					comp_demand_p_input[selected_comp_index][t]=comp_demand_p_input[selected_comp_index][t]+1
					comp_demand[selected_comp_index][t]=comp_demand[selected_comp_index][t]+1
					comp_profit[selected_comp_index][t]=comp_profit[selected_comp_index][t]+prices_this_t[selected_comp_index]
				#else:
					#all competitors are too highly priced for this customer
		#record prices on offer in the current time period
		for c in range(C):
			prices_historical[c][t]=prices_this_t[c]
total_comp_profit=[0 for i in range(C)]
for i in range(C):
    for t in range(T):
        total_comp_profit[i]=total_comp_profit[i]+comp_profit[i][t]
print('full competition results')		
print(total_comp_profit)
		
x=np.linspace(0,1000,1000)
y=comp_profit#
z=prices_historical 
plt.figure(1)  

#competitor_names

prof_plots=[]
for i in range(len(competitor_names)):
	prof_plots.append(plt.plot(x,y[i,:],linewidth=1,label=competitor_names[i]))
	
plt.legend()	#prof_plots[:], competitor_names[:]
plt.title('profit per time period in the last rep')
#rand_prof, = plt.plot(x,y[0,:],linewidth=1)
#fixed_prof, = plt.plot(x,y[1,:],linewidth=1)
#epsilon_prof, = plt.plot(x,y[2,:],linewidth=1)
#####demand_prof_prof, = plt.plot(x,y[3,:])
#demand_prof_exp_smooth, = plt.plot(x,y[3,:],linewidth=1)
#demand_prof_cheap_subset, = plt.plot(x,y[4,:],linewidth=1)
#mode_prof, = plt.plot(x,y[5,:],linewidth=1)
#sine_prof, = plt.plot(x,y[6,:],linewidth=1)
#two_arm_prof, = plt.plot(x,y[7,:],linewidth=1)
#three_arm_prof, = plt.plot(x,y[8,:],linewidth=1)
#epsilon2_prof, = plt.plot(x,y[9,:],linewidth=1)

#plt.legend([rand_prof,fixed_prof,epsilon_prof,demand_prof_exp_smooth,demand_prof_cheap_subset,mode_prof,sine_prof,two_arm_prof,three_arm_prof,epsilon2_prof], ['rand_prof','fixed_prof','epsilon_prof','demand_prof_prof_exp_smooth','demand_prof_prof_cheap_subset','mode_prof','sine_prof','two_arm_prof','three_arm_prof','epsilon2_prof'])

plt.show()


plt.figure(2)

price_plots=[]
for i in range(len(competitor_names)):
	price_plots.append(plt.plot(x,z[i,:],linewidth=1,label=competitor_names[i]))
plt.legend()	#price_plots[:], competitor_names[:]
plt.title('prices per time period in the last rep')


#rand_prof_z, = plt.plot(x,z[0,:])
#fixed_prof_z, = plt.plot(x,z[1,:])
#epsilon_prof_z, = plt.plot(x,z[2,:])
######demand_prof_prof_z, = plt.plot(x,z[3,:])
#demand_prof_exp_smooth_z, = plt.plot(x,z[3,:])
#demand_prof_cheap_subset_z, = plt.plot(x,z[4,:])
#mode_prof_z, = plt.plot(x,z[5,:])
#sine_prof_z, = plt.plot(x,z[6,:])
#two_arm_prof_z, = plt.plot(x,z[7,:]) 
#three_arm_prof_z, = plt.plot(x,z[8,:])
#epsilon2_prof_z, = plt.plot(x,z[9,:])

#plt.legend([fixed_prof_z,epsilon_prof_z,demand_prof_exp_smooth_z,demand_prof_cheap_subset_z,mode_prof_z,sine_prof_z, two_arm_prof_z,three_arm_prof_z,epsilon2_prof_z], ['fixed','epsilon','demand_prof_exp_smooth','demand_prof_cheap_subset','mode','sine','two_arm','three_arm','epsilon2_prof_z'])

plt.show()


##plot graphs to see the result
#time_axes=np.arange(T)
#line_styles=np.chararray((C))
#line_styles[0]='r--'
#line_styles[1]='b--'
#line_styles[2]='g--'
#line_styles[3]='k--'
#line_styles[4]='y--'
#plt.figure(1)
#
##plt.plot(comp_demand[:,0],time_axes,'r--',comp_demand[:,1],time_axes,'b--',comp_demand[:,2],time_axes,'g--',comp_demand[:,3],time_axes,'k--',comp_demand[:,4],time_axes,'y--')
#plt.plot(time_axes,comp_demand[0],'r--',time_axes,comp_demand[1],'b--',time_axes,comp_demand[2],'g--',time_axes,comp_demand[3],'k--',time_axes,comp_demand[4],'y--')
#
##for c in range(C):
#	#plt.plot(comp_demand[:,c],time_axes)
#plt.show()