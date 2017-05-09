#!/usr/bin/env python
import random #see https://docs.python.org/2/library/random.html for lots of the feature of random
import numpy as np
import matplotlib.pyplot as plt

from Fixed_price_competitor import *
from Random_price_competitor import *
from Epsilon_greedy_competitor import *
from demand_profile_competitor import *
from demand_profile_competitor_exp_smooth import *
from demand_model_1 import *
from demand_model_2 import *


#x=np.array([[7,8,5],[3,5,7]],np.int32)
#plt.plot(x[:,0],x[:,1])
#plt.show()

repeats=10;

C=0;#number of competitors, initialised below

T=1000#time periods
N=50#customer population size
arrival_rate=0.5#average arrival rate in each time period from the population of customers 



np.random.seed(0)

#list of object derived from the Competitor class
competitor_objs=[]
#generate competitor objects (the content of parameterdump are stored within these objects in this framework. For the competition version these will be transferred to the parameterdump object (which is not used within this testing framework))
rand_comp_1=Random_price_competitor(0)
fixed_comp_1=Fixed_price_competitor(1, 50)
#epsilon competitor
epsilon=0.1
epsilon_greedy_comp_1=Epsilon_greedy_competitor(2, epsilon)


#demand profile competitor
price_profile_comp_1=demand_profile_competitor(3, np)

#demand profile competitor exponential price profile prediction (with trend)
price_profile_comp_2=demand_profile_competitor_exp_smooth(4, np)

#add competitors to list
competitor_objs.append(rand_comp_1)
competitor_objs.append(fixed_comp_1)
competitor_objs.append(epsilon_greedy_comp_1)
competitor_objs.append(price_profile_comp_1)
competitor_objs.append(price_profile_comp_2)

C=len(competitor_objs)

#DEMAND MODEL INITIALISATION
#model parameters
a=1;
b=1;
#normal willingness to pay distribution
mu=50
sigma=10
#demand model 1
dm_1=demand_model_1(C, a, b, mu, sigma)


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

#repeat runs
for rep in range(repeats):
	print('rep=',rep)
	for c1 in range(C-1):
		for c2 in range(c1+1,C):
			#reset demand array
			comp_demand_p_input=np.zeros((C,T))
			#reset prices_historical
			prices_historical=np.zeros((C,T))
			#the two competitors
			competitor_set=[c1,c2]
			#time steps
			for t in range(T):
				#get competitor prices for the current time period (current/next: check whic for actual competition. In this version competit)
				prices_this_t=[]#prices this time period (to avoid contaminating the timeline)
				for c_number in competitor_set:
					prices_this_t.append(competitor_objs[c_number].p(prices_historical, comp_demand_p_input[c_number], t))
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
						#else:
							#all competitors are too highly priced for this customer
				#record prices on offer in the current time period
				for c in range(2):
					prices_historical[competitor_set[c]][t]=prices_this_t[c]
total_comp_profit=[0 for i in range(C)]
total_comp_profit_matrix=[[0 for j in range(C)] for i in range(C)]
for i in range(C):
    for t in range(T):
        total_comp_profit[i]=total_comp_profit[i]+comp_profit[i][t]
		for j in range(C):
			total_comp_profit_matrix[i][j]=total_comp_profit_matrix[i][j]+comp_profit_matrix[i][j][t]
print('duopoly results')		
print('total_comp_profit(vector)=')		
print(total_comp_profit)
print('total_comp_profit_matrix=')
print(total_comp_profit_matrix)

print('these graphs focus on the overall profit not a pairwise comparison (generate plots from "comp_profit_matrix" for this)')
x=np.linspace(0,1000,1000)
y=comp_profit#
z=prices_historical        
rand_prof, = plt.plot(x,y[0,:])
fixed_prof, = plt.plot(x,y[1,:])
epsilon_prof, = plt.plot(x,y[2,:])
demand_prof_prof, = plt.plot(x,y[3,:])
demand_prof_prof_exp_smooth, = plt.plot(x,y[4,:])

plt.legend([rand_prof,fixed_prof,epsilon_prof,demand_prof_prof,demand_prof_prof_exp_smooth], ['rand_prof','fixed_prof','epsilon_prof','demand_prof_prof','demand_prof_prof_exp_smooth'])

plt.figure(1)

plt.show()

#rand_prof_z, = plt.plot(x,z[0,:])
fixed_prof_z, = plt.plot(x,z[1,:])
epsilon_prof_z, = plt.plot(x,z[2,:])
demand_prof_prof_z, = plt.plot(x,z[3,:])
demand_prof_prof_exp_smooth_z, = plt.plot(x,z[4,:])

plt.legend([fixed_prof_z,epsilon_prof_z,demand_prof_prof_z,demand_prof_prof_exp_smooth_z], ['fixed_prof','epsilon_prof','demand_prof_prof','demand_prof_prof_exp_smooth'])

plt.figure(2)

plt.show()






#plot graphs to see the result
# time_axes=np.arange(T)
# line_styles=np.chararray((C))
# line_styles[0]='r--'
# line_styles[1]='b--'
# line_styles[2]='g--'
# line_styles[3]='k--'
# line_styles[4]='y--'
# plt.figure(1)

# #plt.plot(comp_demand[:,0],time_axes,'r--',comp_demand[:,1],time_axes,'b--',comp_demand[:,2],time_axes,'g--',comp_demand[:,3],time_axes,'k--',comp_demand[:,4],time_axes,'y--')
# plt.plot(time_axes,comp_demand[0],'r--',time_axes,comp_demand[1],'b--',time_axes,comp_demand[2],'g--',time_axes,comp_demand[3],'k--',time_axes,comp_demand[4],'y--')

# #for c in range(C):
	# #plt.plot(comp_demand[:,c],time_axes)
# plt.show()