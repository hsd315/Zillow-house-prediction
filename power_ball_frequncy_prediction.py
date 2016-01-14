# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:02:49 2016

@author: weizhi
"""
import pandas as pd
#data = list(set([8,9,7,22,1,15,19,10,11,12,18,6,29,2,18,1,37,11,26,4,22,16]))
             
data  = pd.read_csv('/Users/weizhi/Desktop/power ball/powerBall_9_30_15_now/Sheet 2-Table 1.csv')
#data = pd.read_csv('/Users/weizhi/Downloads/Powerball.csv')
#data1 = pd.read_csv('/Users/weizhi/Desktop/power ball/powerBall_9_30_15_now/01_2015_09_2015.numbers')

#%%
data = data.drop(['Unnamed: 4'],1)
data = data.fillna(0)


#history= history.drop(['Unnamed: 3'],1)
#history = history.fillna(0)
#%% merge two table

#data['Power Ball frequency'] = data['Power Ball frequency'] + history['3']

#data['White Balls Frequency'] = data['White Balls Frequency'] + history['0']


#%%
result = data.sort(['White Balls Frequency','Power Ball frequency'],ascending=[0,0])



whiteBallFre = data.sort(['White Balls Frequency'],ascending=0)['White Balls Frequency']
whitBall = data.sort(['White Balls Frequency'],ascending=0)['Ball number']


powerBallFre = data.sort(['Power Ball frequency'],ascending=0)['Power Ball frequency']
powerBallFre = data.sort(['Power Ball frequency'],ascending=0)['Ball number']

#%%
import math
count1 = 0
whiteBallMap = data[['Ball number','White Balls Frequency']][:-1]
frequencyMap = {}
for index in range(len(whiteBallMap['Ball number'])):

    key = whiteBallMap['Ball number'][index]
    frequencyMap[key] = whiteBallMap['White Balls Frequency'][index]
    if math.isnan(frequencyMap[key])==False:
        count1 +=whiteBallMap['White Balls Frequency'][index]

#%% power ball 
powerBallMap = data[['Ball number','Power Ball frequency']][:26]
frequencyPowerBall = {}
count = 0
for index in range(len(powerBallMap['Ball number'])):
    key = powerBallMap['Ball number'][index]
    
    frequencyPowerBall[key] = powerBallMap['Power Ball frequency'][index]
    count +=powerBallMap['Power Ball frequency'][index]
#%% power ball calculate
Obj = Solution(frequencyMap,frequencyPowerBall)
num = list(whitBall[:-1].values)
#result2 = Obj.permute(num)
#print result

#%%

class Solution2(object):
    def __init__(self,frequency,frequencyPowerBall):
        self.freqeuncy = frequency 
        self.frequencyPowerBall = frequencyPowerBall
        
    def permute(self,num):
        if len(num)==0:
            return []
        if len(num)==1:
            return [num]
        result = []
        flag = [False for i in range(len(num))]
        self.helper(num,result,flag,[])
        return result
        
    def helper(self,num,result,flag,stk):
        if len(stk) == 5:
            score = self.scoreGet(stk[:])
            
            if score>13:#count1/69*5 = 10

                for item in self.frequencyPowerBall.keys():

                    powerball = self.frequencyPowerBall[item]
                    print powerball
                    if powerball*3 + score>18: # powerball = 1.1153846153846154
                        result.append((stk[:],item,powerball*2 + score,powerball*2)) 
                        # whiteball, readball, totoalsore, powerballsocre
                    print (stk[:],item)
            return result
        for i in range(len(num)):
            if not flag[i]:
                flag[i] = True
                stk.append(num[i])
               # print stk
                self.helper(num,result,flag,stk)
                stk.pop()
                flag[i] = False
    def scoreGet(self,num):
        score = 0
        for key in num:
            score += self.freqeuncy[key]
        return score
Obj = Solution2(frequencyMap,frequencyPowerBall)

import random
num = list(whitBall.values)[:13]
#num = random.sample(num,10)


#%%
result = Obj.permute(num)


data = pd.DataFrame(result)

result = data.sort([2], ascending=0)


resultPower = data.sort([2], ascending=0)

#powerBall =list(set(result[1]))

data.to_csv('/Users/weizhi/Desktop/power ball/result.csv')

#%% score -- 8, 
import random
scoreFinal = sorted(set(data[2]))
count = 20
indexResult = []

for i in range(22,33):
    score = result[result[2] ==i]
    score = score[score[3]>=3]
    if score.shape[0]>20:
        index1 = random.sample(range(score.shape[0]),20)
        print score.iloc[index1]






