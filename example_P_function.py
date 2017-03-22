#!/usr/bin/env python
import random #see https://docs.python.org/2/library/random.html for lots of the feature of random
import math

from future import division
import numpy as np
from sklearn import linear_model
def p(prices_historical, demand_historical, parameterdump=None):
	# if it's the first day
	if demand_historical.size == 0:
		popt = np.random.uniform(0,20)
	# if it's not the first day
	else:
		#fit ols
		regr = linear_model.LinearRegression()
		regr.fit(prices_historical[0].reshape(-1 ,1) ,demand_historical)
		#optimize price and add some noise
		prange = np.linspace(0,10,1000)
		profit = prange * (regr.intercept_ + regr.coef_ * prange)
		popt = prange[np.argmax(profit)] + np.random.normal(0,3)
		#check if price is positive
		if popt < 0 :
			popt = 0.0
		parameterdump = regr
	return (popt ,parameterdump)