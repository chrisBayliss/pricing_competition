from Competitor import Competitor
import math as Math
import numpy as np

class dm1_competitor(Competitor):
	Base_value=[]
	Trend=[]
	prices_next_t=[]
	alpha=0.2
	beta=0.2
	
	#
	initB=2
	initSIGMA=10
	initMU=50
	
	B=2
	SIGMA=10
	MU=50
	
	WTPSS=100
	ZScores=[-2.5724655387049906,-2.1684599309842865,-1.9590489380232072,-1.811392700417685,-1.6951255314671452,-1.5980814841640125,-1.5140978937571379,-1.4396003356430906,-1.372321668169808,-1.3107291349449381,-1.2537355211790808,-1.2005401847480213,-1.1505354841879978,-1.1032486849492416,-1.058304292774104,-1.015398735127869,-0.9742828165154709,-0.9347492395472065,-0.8966235262394615,-0.8597572811063086,-0.8240231038913786,-0.7893106877951082,-0.7555237849300253,-0.7225778163747846,-0.6903979682827458,-0.6589176592955888,-0.6280772949785438,-0.5978232465333634,-0.5681070065006705,-0.5388844854043503,-0.5101154215692982,-0.4817628825109444,-0.4537928409385943,-0.42617381194541953,-0.39887654066729156,-0.371873731789539,-0.3451398139158496,-0.3186507330984484,-0.2923837708445634,-0.2663173827224585,-0.24043105433669845,-0.21470517196169633,-0.18912090554128483,-0.1636601021005565,-0.13830518789028187,-0.113039077805617,-0.08784509079950176,-0.06270687015391799,-0.03760830758515286,-0.01253347024610061,0.01253347024610061,0.03760830758515286,0.06270687015391799,0.08784509079950176,0.113039077805617,0.13830518789028187,0.1636601021005565,0.18912090554128483,0.21470517196169633,0.24043105433669845,0.2663173827224585,0.2923837708445634,0.3186507330984484,0.3451398139158496,0.371873731789539,0.39887654066729156,0.42617381194541953,0.4537928409385943,0.4817628825109444,0.5101154215692982,0.5388844854043503,0.5681070065006705,0.5978232465333634,0.6280772949785438,0.6589176592955888,0.6903979682827458,0.7225778163747846,0.7555237849300253,0.7893106877951082,0.8240231038913786,0.8597572811063086,0.8966235262394625,0.9347492395472065,0.9742828165154709,1.015398735127869,1.058304292774104,1.1032486849492416,1.1505354841879978,1.2005401847480213,1.2537355211790808,1.3107291349449381,1.3723216681698085,1.439600335643091,1.5140978937571383,1.5980814841640127,1.6951255314671456,1.8113927004176849,1.9590489380232072,2.1684599309842865,2.5724655387049906]

	
	def __init__(self, competitor_number, np):
		Competitor.__init__(self, competitor_number)
		
		self.competitor_number=competitor_number
		self.np=np
		
		
		
		unitarySumOfDecreasingStepLengths=0
		for i in range(self.iterations_2*1000):
			unitarySumOfDecreasingStepLengths=unitarySumOfDecreasingStepLengths+(1-(i/self.iterations_2*1000))
		
		self.initial_step_lengths_2=np.zeros((self.parameters_2))
		for i in range(self.parameters_2):
			self.initial_step_lengths_2[i]=(self.times_across_space_2*self.max_param_ranges_2[i])/(self.parameter_selection_distribution_2_step_len[i]*unitarySumOfDecreasingStepLengths)
			#self.initial_step_lengths_2[i]=0.01;
			
			
	def p(self, prices_historical, demand_historical, t):#, parameterdump
		
		prev_time_periods_used_in_linear_regression=50
		PPU=prev_time_periods_used_in_linear_regression
		
		# if it's the first day
		#if demand_historical.size == 0:
		if t == 0:
			#store the number of competitors parameter in parameterdump
			self.C=len(prices_historical)
			
			self.Base_value=[[0 for j in range(self.C)] for i in range(self.T)]
			self.Trend=[[0 for j in range(self.C)] for i in range(self.T)]
			
			self.prices_next_t=[0 for i in range(self.C)]
		elif t<2:
			#initialise base value (trend initialised to 0)
			#for c in range(self.C):
			[sorted_prices, ind_order]=self.sort(prices_historical[:,t-1])
			self.Base_value[0]=sorted_prices
			
			#random initial price
			popt = np.random.uniform(0,100)
		else:
			#update demand model parameters (correction based on observations)
			
			######################################################
			#THE CURRENT DEMAND PROFILE FITTING MODEL IS A SEVERE BOTTLE NECK
			#MAYBE MULTI-ARMED BANDITS CAN BE USED TO SEARCH THE DEMAND MODEL SPACE
			initial_parameters_2=[self.MU, self.SIGMA, self.B, self.N]
			
			#current ones are used as the initial solution
			updated_params_2=self.simulatedAnnealing_2(self.iterations_2, self.parameters_2, initial_parameters_2, self.initial_step_lengths_2, self.parameter_selection_distribution_2, self.param_bounds_2, self.t0Factor_2, prices_historical, demand_historical, t, self.np)
			
			[self.MU, self.SIGMA, self.B, self.N]=updated_params_2
			#####################################################
			
			#prices offered in the previous time period
			[sorted_prices, ind_order]=self.sort(prices_historical[:,t-1])
			
			
			forecast_price_set=self.update_exp_smooth_params_return_forecast_prices(sorted_prices, t)
			#sample wtp and derive competitor probability distribution
			#find optimum d*p
			#print(ind_order)
			us_index=0
			while ind_order[us_index]!=self.competitor_number:
				us_index=us_index+1
			#print(us_index)
			popt = self.next_price(forecast_price_set, us_index)
		
		return popt