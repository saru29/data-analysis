# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 15:01:22 2022

@author: Jay
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# read csv 
df= pd.read_csv('Video_GamesDec_2016 .csv')
sales = df.dropna(subset=['Name','Year_of_Release','Genre','Publisher'])
scores=df.dropna()
#export cleaned datasets to csv
sales.to_csv('vgsales2016.csv', index=False)
scores.to_csv('vgscores2016.csv',index=False)

plt.figure(figsize=(10,7))  
sns.set(font_scale=2)
scores["User_Score"] = scores["User_Score"].astype('float')
ax = sns.violinplot(x='Platform', y='User_Score', data=scores, order=['Wii','PS2','X360','PS3','PS4','XB'])
ax.set(xlabel='Console', ylabel='User Score', title='Best-Selling Consoles: User Scores')
plt.show()

df2= pd.read_csv('vgsales_2017.csv')
df2.isnull().sum()
scores2 = df2.dropna(subset=['Year','Publisher'])
scores2.isnull().sum()
#export cleaned datasets to csv
scores2.to_csv('vgscores2017.csv',index=False)

plt.figure(figsize=(10,7))  
sns.set(font_scale=2)
scores2["User_Score"] = scores["User_Score"].astype('float')
ax = sns.violinplot(x='Platform', y='User_Score', data=scores, order=['Wii','PS2','X360','PS3','PS','XB','PS4'])
ax.set(xlabel='Console', ylabel='User Score', title='Best-Selling Consoles: User Scores')
plt.show()

df3= pd.read_csv('vgsales-12-4-2019-short.csv')
df3.rename(columns={"PAL_Sales":"EU_Sales"},inplace=True)
df3.isnull().sum()
df3.drop('Total_Shipped', axis=1, inplace=True)
scores3= df3.dropna(subset= ['Global_Sales'])
scores3.isnull().sum()
scores3.to_csv(vgscores2019.csv,index=False)