# -*- coding: utf-8 -*-	
"""
Created on Mon Mar 06 12:03:52 2017

@author: ae1g15
"""
from Competitor import Competitor
import os, math, operator, random, csv, scipy
import numpy as np

class Epsilon_greedy_competitor(Competitor):	



	#initialization
	epsilon=0.2 #probability that a random price is chosen
	counts = [0 for col in range(10)] # a vector with 10 entries showing how many times each price is chosen
	values = [0.0 for col in range(10)] # a vector with 10 entries showing the average reward(demand) obtained by each price
	index_last_period=0

	#the code is for epsilon-greedy
	#in epsilon-decreasing epsilon=min{1,epsilon0/t}

	prices=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100] #we test these 10 prices to compare their demand 

	def __init__(self, competitor_number, epsilon):
			Competitor.__init__(self, competitor_number)
			self.epsilon=epsilon
		
	def p(self, prices_historical, demand_historical, t):#, parameterdump
		index=-1#initialise 'index'
		if t==0: #if it is the first day
			#store the number of competitors parameter in parameterdump
			self.C=prices_historical.size
			self.reset()
			index=random.randrange(len(self.values)) #choose a random price
		else: #if it's not the first day 
                  #update for the demand in the previos time period
			#print(demand_historical[t-1],',',self.prices[self.index_last_period])  
			self.values[self.index_last_period]=self.update(self.index_last_period,demand_historical[t-1]*self.prices[self.index_last_period])
			index=self.select_arm(self.epsilon,self.values) #index is the price chosen at t
			
			
		#update last index ready for next time period
		self.index_last_period=index
		#print(self.values)  
		self.counts[index]=self.counts[index]+1 #the number of times this price is chosen increases by 1
		
		#after we have chosen price index we receive the demand
		
		
		
		popt = self.prices[index]
		return popt

	def ind_max(self, x):
		m = max(x)
		return x.index(m)

	def select_arm(self, epsilon,values): #it returns which arm to play
		rand_num=np.random.uniform(0,1)
		#print(rand_num,',',epsilon)  
		if rand_num > epsilon: #it chooses randomly whether it will explore or exploit
			return self.ind_max(values) #it selects the price with the highest demand so far
		else:
			arm=random.randrange(len(values))
			#print(arm) 
			return arm #it selects a random arm
   
			
	def update(self, chosen_arm, reward):
		#print(chosen_arm)
		self.counts[chosen_arm] = self.counts[chosen_arm] + 1
		n = self.counts[chosen_arm]
		value = self.values[chosen_arm]
		new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward #weighted average
		return new_value


	def reset(self):
         	self.counts = [0 for col in range(10)] # a vector with 10 entries showing how many times each price is chosen
         	self.values = [0.0 for col in range(10)] # a vector with 10 entries showing the average reward(demand) obtained by each price
         	self.index_last_period=0    