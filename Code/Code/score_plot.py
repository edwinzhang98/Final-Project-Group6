#%%
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
os.getcwd()
os.chdir("../score_save")
print(os.getcwd())
df_CNN = pd.read_csv('CNN_epoch30_batch64.csv').drop(columns='Unnamed: 0')
df_Resnet18 = pd.read_csv('resnet18_lr3e-4.csv').drop(columns='Unnamed: 0')
df_VIT = pd.read_csv('VIT_lr1e-3.csv').drop(columns='Unnamed: 0')

metrics_list = df_CNN.columns
#%%
f, (ax1,ax2)=plt.subplots(1,2,figsize=(20,8))
ax1.plot(df_CNN['Valid loss'])
ax1.plot(df_Resnet18['Valid loss'])
ax1.plot(df_VIT['Valid loss'])
ax1.legend(['CNN','Resnet18','Vision Transformer'])
ax1.set_title('Validation loss')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')

ax2.plot(df_CNN['Train loss'])
ax2.plot(df_Resnet18['Train loss'])
ax2.plot(df_VIT['Train loss'])
ax2.legend(['CNN','Resnet18','Vision Transformer'])
ax2.set_title('Training loss')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss')
plt.show()
