#/usr/bin/env python
import random #see https://docs.python.org/2/library/random.html for lots of the feature of random
import math
#from future import division
import numpy as np
from sklearn import linear_model

#derive different competitors from this class and overwrite the P function
class Competitor(object):

	T=1000
	C=0
	price_history=np.zeros((T))

	competitor_number=0;


	#parameterdump=parameterdump()

	def __init__(self, competitor_number):
		self.competitor_number=competitor_number
	




	def p(self, prices_historical, demand_historical):#, parameterdump
    
    
      t0=100
      current_time = demand_historical, size
    #If it's the first time period
    if current_time==0:
        #Set up the prices to charge during the initialisation phase
        #Define array to store whether value has been used
        used=np.zeros((t0))
        init_prices=np.zeros((t0))
        for t in range t0-1:
            #Sample with replacement from each of the t0 ranges between 0 and 100
            while True:
                chosen_range= random.randint(0,100) #Assumes max Price 100 in LHS 
                #Check if we've had it before
                if used[chosen_range]==0:
                    #Haven't used it before so sample from within this range
                    init_prices[t]=random.uniform(chosen_range,chosen_range+1/t0)#NB need to pass this back in somehow - haven't worked that out!
                    break
                
    else if current_time <=t0:
        #Charge price calculated during initialisation
        return init_prices[current_time]
	else:
        #Run the main method
        #Split into duopoly versus full market
        num_competitors = prices_historical, size #??? Not sure if this is going to get the right dimension!!
              
        if num_competitors == 1:
            #In the round robin phase
            #Forecast competitor's next price using exponential smoothing
            pred_comp_price = 50 #TEMPORARY until can put in link to exp smoothing
            
            #Fit each of the demand models
            
            #Output a measure of revenue/quality of fit for each demand model as a way of choosing which one
            
            #Select which arm to pull out of num_Arms
            
            #Find optimal price associated with the chosen arm
            opt_Price = 50 #TEMPORARY
           
        else
            #In the full market phase
            #Forecast the comopetitors' modal price using exponential smoothing
            pred_comp_price=50 #TEMPORARY until can put a link into exponential smoothing
            #NOT SURE WHETHER THE FOLLOWING CAN BE THE SAME FOR BOTH ROUND ROBIN AND FULL MARKET. IF SO, TAKE OUT OF LOOP.
            #Fit each of the demand models
            
            #Output a measure of revenue/quality of fit for each demand model as a way of choosing which one
            
            #Select which arm to pull out of num_Arms
        
         #Set up random perturbation
         power=1/6
         a_t = 1/current_time^power
         #Find slope Rev_predicted(p_opt+c_t)-Rev_predicted(p_opt-c_t)
         rev_Slope=0.1 #TEMPORARY
         #Perturb optimal price by a small amount
         perturbation=(a_t/2*c_t)*rev_Slope
        if opt_price+perturbation > 0 then:
            opt_price=opt_price+perturbation
            
        #Put init_prices and current values of fits into parameterdump
        
        return opt_Price, parameterdump 
            
            
        
		
   