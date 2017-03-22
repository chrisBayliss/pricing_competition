#!/usr/bin/env python
import random #see https://docs.python.org/2/library/random.html for lots of the feature of random
import numpy as np
import matplotlib.pyplot as plt


#x=np.array([[7,8,5],[3,5,7]],np.int32)
#plt.plot(x[:,0],x[:,1])
#plt.show()



repeats=10;

C=5;#number of competitors

T=1000#time periods
N=50#customer population size
arrival_rate=0.5#average arrival rate in each time period from the population of customers 

#model parameters
a=1;
b=1;

#normal willingness to pay distribution
mu=50
sigma=10

np.random.seed(0)

s=np.random.normal(mu, sigma, 10)
#u=np.random.normal(mu, sigma, 10,10)

x=np.array([[7,8,5],[3,5,7]],np.int32)
for i in range(5):
	print(i)
	print(x[1][1])
	print(x)
	
print('next')

#print(u)
#for i in 10:
	#for j in 10:
		#print(u[i][j])

for i in range(len(s)):
	print(s[i])

#non-dynmaic randomly generated customer prices
comp_prices=np.zeros((T,C))
for t in range(T):
	comp_prices[t]=np.random.normal(mu, sigma, C)
#print(comp_prices)

#demand per competitor in each time period
comp_demand=np.zeros((T,C))
#total profit per competitor
comp_profit=np.zeros((T,C))

#initialise array for the customer competitor cumulative distribution
cust_comp_select_cumu_dist=np.zeros((C))
sum_cumu_dist=0
selected_comp_index=0

#repeat runs
for rep in range(repeats):
	print('rep=',rep)
	#time steps
	for t in range(T):
		#customer arrival process
		for k in range(N):
			rand_arrival_number=np.random.uniform(0,1,1)
			if rand_arrival_number<arrival_rate:
				#print('hello',rand_arrival_number)
				#generate random willingness to pay
				cust_w_t_p=np.random.normal(mu, sigma, 1)
				#generate (cumulative) probability distribution for competitor selection (0 if price>= w.t.p)
				sum_cumu_dist=0
				for c in range(C):
					if comp_prices[t][c]<cust_w_t_p:
						sum_cumu_dist=sum_cumu_dist+a*((cust_w_t_p-comp_prices[t][c])/cust_w_t_p)**b
						cust_comp_select_cumu_dist[c]=sum_cumu_dist
					else:
						cust_comp_select_cumu_dist[c]=sum_cumu_dist
						
				#make the probabilities sum to 1
				if sum_cumu_dist>0:
					for c in range(C):
						cust_comp_select_cumu_dist[c]=cust_comp_select_cumu_dist[c]/sum_cumu_dist
					#print(cust_comp_select_cumu_dist)
					
					#generate a random number to select a competitor to buy the product from
					rand_comp_selection_number=np.random.uniform(0,1,1)
					#print(rand_comp_selection_number)
					selected_comp_index=0
					while rand_comp_selection_number>=cust_comp_select_cumu_dist[selected_comp_index]:
						selected_comp_index=selected_comp_index+1
					#update the demand and profit of the competitor who won the customers business
					comp_demand[t][selected_comp_index]=comp_demand[t][selected_comp_index]+1
					comp_profit[t][selected_comp_index]=comp_profit[t][selected_comp_index]+comp_prices[t][selected_comp_index]
				#else:
					#all competitors are too highly priced for this customer
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
plt.plot(time_axes,comp_demand[:,0],'r--',time_axes,comp_demand[:,1],'b--',time_axes,comp_demand[:,2],'g--',time_axes,comp_demand[:,3],'k--',time_axes,comp_demand[:,4],'y--')

#for c in range(C):
	#plt.plot(comp_demand[:,c],time_axes)
plt.show()