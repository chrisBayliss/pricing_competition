# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:47:38 2017

@author: Asus
"""

from Competitor import Competitor
import os, math, operator, random, csv, scipy
import numpy as np
import Math

class Epsilon_greedy_competitor(Competitor):	



	#initialization
	epsilon=0.2 #probability that a random price is chosen
	index_last_period=0
	selection_matrix=np.zeros(shape=(10,10))
	counts=np.zeros(shape=(10,10))
	previous_mode_price_index=0
    
    
    
    
    
    
	Base_value=[]
	Trend=[]
	prices_next_t=[]
	alpha=0.2
	beta=0.2
	
	#mode_intervals=[10,20,30,40,50,60,70,80,90,100]
	mode_interval_size=10
	mode_interval_frequencies=[0]*10
	
	other_prices=[]
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
			self.C=len(prices_historical)
			
			self.Base_value=[[0 for j in range(self.C-1)] for i in range(self.T)]
			self.Trend=[[0 for j in range(self.C-1)] for i in range(self.T)]
			
			self.prices_next_t=[0 for i in range(self.C-1)]
			
			self.other_prices=[0]*(self.C-1)
			
			popt=random.randrange(len(self.values)) #choose a random price
		
        
		else: #if it's not the first day 
            
			
                  #update for the demand in the previos time period
			#print(demand_historical[t-1],',',self.prices[self.index_last_period])  
			counter=0
			for i in range(self.C):
				if i!=self.competitor_number:
					self.other_prices[counter]=prices_historical[i,t-1]
					counter=counter+1		
			[sorted_prices, ind_order]=self.sort(self.other_prices)#prices_historical[:,t-1]				            
			forecast_price_set=self.update_exp_smooth_params_return_forecast_prices(sorted_prices, t)
			if t==1:
			     self.Base_value[0]=sorted_prices
                       
                       
           #update step
			for i in range(len(self.mode_interval_frequencies)):
				self.mode_interval_frequencies[i]=0
				
			for i in range(len(self.other_prices)):
				interval=min(len(self.mode_interval_frequencies)-1, max(0, Math.floor(self.other_prices[i]/self.mode_interval_size)))
				self.mode_interval_frequencies[interval]=self.mode_interval_frequencies[interval]+1
                                  
			 #find the mode price
			max_frequency=0
			mode_price_index=None
			for i in range(len(self.mode_interval_frequencies)):
				if self.mode_interval_frequencies[i]>max_frequency:
					max_frequency=self.mode_interval_frequencies[i]
					mode_price_index=i            
                    
                    
			self.selection_matrix[mode_price_index, self.index_last_period]=self.update(self.index_last_period,demand_historical[t-1]*self.prices[self.index_last_period])         
        
                       
           #price optimisation
           ################            
           #mode interval frequencies
			#reset
			for i in range(len(self.mode_interval_frequencies)):
				self.mode_interval_frequencies[i]=0
				
			for i in range(len(forecast_price_set)):
				interval=min(len(self.mode_interval_frequencies)-1, max(0, Math.floor(forecast_price_set[i]/self.mode_interval_size)))
				self.mode_interval_frequencies[interval]=self.mode_interval_frequencies[interval]+1
			
			#find the mode price
			max_frequency=0
			forecast_mode_price_index=None
			for i in range(len(self.mode_interval_frequencies)):
				if self.mode_interval_frequencies[i]>max_frequency:
					max_frequency=self.mode_interval_frequencies[i]
					forecast_mode_price_index=i 
            
			
			index=self.select_arm(self.epsilon,self.values[forecast_mode_price_index]) #index is the price chosen at t
			
			
		#update last index ready for next time period
		self.index_last_period=index
		#print(self.values)  
		self.counts[index]=self.counts[index]+1 #the number of times this price is chosen increases by 1
		
		#after we have chosen price index we receive the demand
		
		
		
		popt = self.prices[index]
		return popt
      
	def update_exp_smooth_params_return_forecast_prices(self, comp_prices_last_t, t):
		   for c in range(self.C-1):
			   self.Base_value[t-1][c]=(self.alpha*comp_prices_last_t[c])+((1-self.alpha)*(self.Base_value[t-2][c]+self.Trend[t-2][c]))			
			   self.Trend[t-1][c]=(self.beta*(self.Base_value[t-1][c]-self.Base_value[t-2][c]))+((1-self.beta)*self.Trend[t-2][c])			
			   self.prices_next_t[c]=max(0, min(100,self.Base_value[t-2][c]+self.Trend[t-1][c]))
		   return self.prices_next_t
    
    
    
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
   
			
	def update(self,row, chosen_arm, reward):
		#print(chosen_arm)
		self.counts[row, chosen_arm] = self.counts[row,chosen_arm] + 1
		n = self.counts[row,chosen_arm]
		value = self.selection_matrix[row,chosen_arm]
		new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward #weighted average
		return new_value


	def reset(self):
         	self.counts = [[0.0 for row in range(10)] for col in range(10)] # a vector with 10 entries showing how many times each price is chosen
         	self.values = [[0.0 for row in range(10)] for col in range(10)]  # a vector with 10 entries showing the average reward(demand) obtained by each price
         	self.index_last_period=0  