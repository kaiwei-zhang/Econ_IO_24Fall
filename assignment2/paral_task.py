import os
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings("ignore")

os.chdir('/Users/kaiweizhang/Documents/GitHub/Econ_IO_24Fall/assignment2')
global MARKET_DATA, CONSUMER, THETA, delta_guess, P_lis
delta_guess = [0]*50
P_lis=[]
for i in range(1,51):
    P_lis.append('P_'+str(i))
THETA = [-0.5, 2, 2] #This is theta_2
MARKET_DATA=pd.read_pickle('MARKET_DATA.pkl')
CONSUMER=pd.read_pickle('CONSUMER.pkl')


def calculate_s(delta_guess,theta, Temp_T=1, market_data=MARKET_DATA, CONSUMER=CONSUMER):
    """
    Given delta_guess and theta, calculate s(prob that i choose j) for each consumer 
    Seems input is only for Temp_T, I need Temp_T to pin down the market data
    return: CONSUMER.query(t), with 50 new columns which tells the probability of choosing each product
    """
    delta_input =np.array([0]+list(delta_guess))

    [alpha, sigma1,sigma2] =theta

    data1=market_data[['product_ID','price','sugar','caffeine','t']]
    # generate a table with price_1,price_2,..price_10 for each t

    temp_data=data1[data1['t']== Temp_T].drop(columns=['t'])
    temp_CONSUMER = CONSUMER[CONSUMER['t']== Temp_T]
    temp_data # data on t=1 market


    p_row = temp_data[['price']].T #50x1 for 50 ID
    sug_row = temp_data[['sugar']].T
    caf_row = temp_data[['caffeine']].T
    table1=pd.concat([p_row,sug_row,caf_row],axis=0)
    table1

    table2=temp_CONSUMER[['income','nu_1','nu_2']]
    table2['income']=table2['income']*alpha
    table2['nu_1']=table2['nu_1']*sigma1
    table2['nu_2']=table2['nu_2']*sigma2
    table2.shape, table1.shape #calculate table1*table2 in matrix
    m1=np.array(table1)
    m2=np.array(table2)
    m = m2@m1
    m.shape  # now I want CONSUMER add m as a new column
    mu_lis=[]

    for i in range(1,51):
        temp_CONSUMER['mu_'+ str(i)] = m[:,i-1]
        mu_lis.append('mu_'+ str(i))
    temp_CONSUMER

    sum_lis = []
    for i in range(1,51):
        temp_CONSUMER['exp_delta_plus_mu'+str(i)] = np.exp(temp_CONSUMER['mu_'+str(i)]+delta_input[i])
        sum_lis.append('exp_delta_plus_mu'+str(i))
    temp_CONSUMER['new'] = temp_CONSUMER[sum_lis].sum(axis=1)+1 #sum_k(exp(delta_k+mu_k))+1
    temp_CONSUMER['new'] = np.log(temp_CONSUMER['new'])

    P_lis=[]
    for i in range(1,51):
        temp_CONSUMER['P_'+str(i)] = temp_CONSUMER['mu_'+str(i)]+delta_input[i]-temp_CONSUMER['new']
        temp_CONSUMER['P_'+str(i)]=np.exp(temp_CONSUMER['P_'+str(i)])
        P_lis.append('P_'+str(i))
    temp_CONSUMER[P_lis].sum(axis=1) # sum to around 1, good

    # drop mu_lis and sum_lis
    temp_CONSUMER=temp_CONSUMER.drop(columns=mu_lis)
    temp_CONSUMER=temp_CONSUMER.drop(columns=sum_lis)
    temp_CONSUMER = temp_CONSUMER.drop(columns='new')


    return temp_CONSUMER

# return abs norm
def calc_diff(a):
    loss = 0
    for i in list(a):
        loss+=abs(i)
    return loss

def predict_share_from_delta_theta(delta_guess,theta, Temp_T=1, market_data = MARKET_DATA, consumer=CONSUMER):
    temp_consumer_data = calculate_s(delta_guess,theta, Temp_T,market_data, consumer)
    ans = temp_consumer_data[P_lis].sum()/len(temp_consumer_data)
    return ans

def solve_delta_from_theta(theta, t, market_data = MARKET_DATA, consumer=CONSUMER, tol=1e-4):
    """Given theta and t, start from delta_guess, update delta_t(theta) until converge"""
    global delta_guess
    delta_late = delta_guess
    diff = 1000
    while diff>tol:
        s_hat = list(predict_share_from_delta_theta(delta_late,theta, Temp_T=t,market_data=market_data,consumer=consumer))
        real_s = list(market_data[market_data['t']==t]['market_share'])
        delta_new = delta_late + np.log(real_s)-np.log(s_hat)
        diff = calc_diff(delta_new-delta_late)
        delta_late = delta_new
    return delta_late

from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")

import time
start_time=time.time()
pd.options.mode.chained_assignment = None
pd.options.mode.copy_on_write = True
delta_long = Parallel(n_jobs=4)(delayed(solve_delta_from_theta)(THETA,t=i) for i in range(1,51))
save=pd.DataFrame()
save['delta']=delta_long
save.to_pickle('delta.pkl')
end_time=time.time()
print(end_time- start_time)