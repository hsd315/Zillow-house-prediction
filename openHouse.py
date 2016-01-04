# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 21:41:14 2016

@author: weizhi
"""

import pandas as pd

pleasanton = pd.read_csv('/Users/weizhi/Downloads/redfin_2016-01-02-21-40-27_results.csv')

family = pleasanton[pleasanton['HOME TYPE']=='Single Family Residential']

price = family['LIST PRICE']

#%% 

import pylab as plt
plt.figure()
plt.plot(price)

#%% all the house

allData = pd.read_csv('/Users/weizhi/Downloads/redfin_2016-01-02-21-52-46_results.csv')
openHouse = allData[allData['NEXT OPEN HOUSE DATE']=='2016-01-03']

family = openHouse[openHouse['HOME TYPE']=='Single Family Residential']

house = family[family['LIST PRICE']<=1200000]
house.to_csv('/Users/weizhi/Downloads/redfin_2016-01-02-21-52-46_results_dublin.csv')

#%% fremond 

allData = pd.read_csv('/Users/weizhi/Downloads/redfin_2016-01-02-22-13-26_results.csv')
openHouse = allData[allData['NEXT OPEN HOUSE DATE']=='2016-01-03']

family = openHouse[openHouse['HOME TYPE']=='Single Family Residential']

house = family[family['LIST PRICE']<=1200000]
house['URL (SEE http://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)'].iloc[0]

house.to_csv('/Users/weizhi/Downloads/redfin_2016-01-02-22-13-26_results_fremont_1M2.csv')
#%% san ramon

#%% fremond 

allData = pd.read_csv('/Users/weizhi/Downloads/redfin_2016-01-02-23-27-28_results.csv')
openHouse = allData[allData['NEXT OPEN HOUSE DATE']=='2016-01-03']

family = openHouse[openHouse['HOME TYPE']=='Single Family Residential']

house = family[family['LIST PRICE']<=1000000]
#house['URL (SEE http://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)'].iloc[0]

house.to_csv('/Users/weizhi/Downloads/redfin_2016-01-02-22-13-26_results_san_ramon_1M2.csv')

