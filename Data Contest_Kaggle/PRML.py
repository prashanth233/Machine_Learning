#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy as sp
import numpy as np
import pandas as pd


# In[2]:


train = pd.read_csv('train.csv')


# In[3]:


data = train.pivot(index=['song_id'], columns=['customer_id'])


# In[4]:


copy = data.fillna(-1,inplace=False)
copy[copy>=0]=1
copy[copy==-1]=0
copy2 = data.fillna(0,inplace=False)


# In[5]:


data=data.fillna(data.mean())


# In[6]:


np.seterr(divide='ignore',invalid='ignore')
temp=data.values
temp=(temp - temp.mean(axis=0))/(temp.max(axis=0)+0.0000001-temp.min(axis=0)) [None,:]
norm = pd.DataFrame(temp, index=data.index, columns=data.columns)
norm[norm*norm<0.000000000001]=0
temp=None
del(train)


# In[7]:


from scipy.spatial.distance import cdist
mat=norm.to_numpy()
sim=pd.DataFrame(1. - cdist(mat,mat,'cosine'),index=norm.index,columns=norm.index)
mat=None
del(norm)


# In[8]:


sim[sim<0]=0


# In[9]:


a=copy2.to_numpy()
b=copy.to_numpy()
c=sim.to_numpy()
del(copy2)
del(copy)
del(sim)


# In[10]:


np.seterr(divide='ignore',invalid='ignore')
d=np.divide(np.matmul(c,a),np.matmul(c,b))
final=pd.DataFrame(d,columns=data.columns,index=data.index)
a=None
b=None
c=None
d=None
del(data)


# In[11]:


final.columns=final.columns.droplevel(0)
final.columns.name=None
final=final.reset_index()


# In[12]:


final1=final.melt(id_vars=['song_id'],var_name='customer_id',value_name='score')
del(final)


# In[13]:


test = pd.read_csv('test.csv')


# In[14]:


res=test.merge(final1, how='inner', left_on=['customer_id', 'song_id'], right_on=['customer_id', 'song_id'])


# In[15]:


res=res.drop(['customer_id', 'song_id'], axis=1)
res=res.fillna(3.93510)
res.to_csv(r'./CS18B013_CS18B004.csv', index_label="test_row_id")

