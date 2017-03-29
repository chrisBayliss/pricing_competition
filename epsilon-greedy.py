# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 12:03:52 2017

@author: ae1g15
"""

import os, math, operator, random, csv, scipy
import numpy as np



#the code is for epsilon-greedy
#in epsilon-decreasing epsilon=min{1,epsilon0/t}
    
prices=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100] #we test these 10 prices to compare their demand   
  
def ind_max(x):
    m = max(x)
    return x.index(m)
    
def select_arm(epsilon,values): #it returns which arm to play
    if random.random() > epsilon: #it chooses randomly whether it will explore or exploit
          return ind_max(values) #it selects the price with the highest demand so far
    else:
          return random.randrange(len(values)) #it selects a random arm
def update(chosen_arm, reward):
     counts[chosen_arm] = counts[chosen_arm] + 1
     n = counts[chosen_arm]
     value = values[chosen_arm]
     new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward #weighted average
     return new_value
 

#initialization
prices_historical=np.zeros(1000) #the price chosen at each time step
demand_historical=np.zeros(1000) #demand obtained at each time step 
epsilon=0.1 #probability that a random price is chosen
counts = [0 for col in range(10)] # a vector with 10 entries showing how many times each price is chosen
values = [0.0 for col in range(10)] # a vector with 10 entries showing the average reward(demand) obtained by each price
for t in range (1000): 
   if t==0: #if it is the first day
       index=random.randrange(len(values)) #choose a random price
   else: #if it's not the first day       
       index=select_arm(epsilon,values) #index is the price chosen at t
   counts[index]=counts[index]+1 #the number of times this price is chosen increases by 1
   prices_historical[t]=prices[index]
   #after we have chosen price index we receive the demand
   values[index]=update(index,demand)
