#!/usr/bin/env python
import random #see https://docs.python.org/2/library/random.html for lots of the feature of random
import numpy as np
import matplotlib.pyplot as plt

from Fixed_price_competitor import *
from Random_price_competitor import *
from demand_model_1 import *


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

competitor_objs.append(rand_comp_1)
competitor_objs.append(fixed_comp_1)

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

#initialise array for the customer competitor cumulative distribution
cust_comp_select_cumu_dist=np.zeros((C))
sum_cumu_dist=0
selected_comp_index=0

#repeat runs
for rep in range(repeats):
	print('rep=',rep)
	#reset demand array
	comp_demand=np.zeros((C,T))
	#reset prices_historical
	prices_historical=np.zeros((C,T))
	for c1 in range(C-1):
		for c2 in range(c1+1,C):
			#the two competitors
			competitor_set=[c1,c2]
			#time steps
			for t in range(T):
				#get competitor prices for the current time period (current/next: check whic for actual competition. In this version competit)
				prices_this_t=[]#prices this time period (to avoid contaminating the timeline)
				for c_number in competitor_set:
					prices_this_t.append(competitor_objs[c_number].p(prices_historical, comp_demand[c_number], t))
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
							comp_demand[competitor_set[selected_comp_index]][t]=comp_demand[competitor_set[selected_comp_index]][t]+1
							comp_profit[competitor_set[selected_comp_index]][t]=comp_profit[competitor_set[selected_comp_index]][t]+prices_this_t[selected_comp_index]
						#else:
							#all competitors are too highly priced for this customer
				#record prices on offer in the current time period
				for c in range(2):
					prices_historical[competitor_set[c]][t]=prices_this_t[c]
#plot graphs to see the result
time_axes=np.arange(T)
line_styles=np.chararray((C))
line_styles[0]='r--'
line_styles[1]='b--'
line_styles[2]='g--'
line_styles[3]='k--'
line_styles[4]='y--'
plt.figure(1)

#plt.plot(comp_demand[:,0],time_axes,'r--',comp_demand[:,1],time_axes,'b--',comp_demand[:,2],time_axes,'g--',comp_demand[:,3],time_axes,'k--',comp_demand[:,4],time_axes,'y--')
plt.plot(time_axes,comp_demand[0],'r--',time_axes,comp_demand[1],'b--',time_axes,comp_demand[2],'g--',time_axes,comp_demand[3],'k--',time_axes,comp_demand[4],'y--')

#for c in range(C):
	#plt.plot(comp_demand[:,c],time_axes)
plt.show()