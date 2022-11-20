#!/usr/bin/env python
# coding: utf-8

# In[103]:


import numpy as np
import pandas as pd


# In[2]:


#Datatype and attributes
#Main datatype- ndarray
a1=np.array([1,2,3,4,5,6])
a1


# In[3]:


type(a1)


# In[17]:


a2=([[1,2,3],[2,3,4]])
a3=([[1,2,3],[2,3,4],[4,5,6]],[[65,23,1],[31,54,12],[23,43,65]])
a2


# In[19]:


import pandas as pd
df=pd.DataFrame(a3)
df


# In[20]:


#Creating arrays
sample_array=np.array([1,2,3])
sample_array


# In[21]:


sample_array.dtype


# ones=np.ones([2,2])
# ones

# In[33]:


ones=np.ones([2,2]) 
ones


# In[36]:


zero=np.zeros([6,65])
zero


# In[38]:


range=np.arange(0,10,2)
range


# In[10]:


random=np.random.randint(0,10, size=(3,4))
random


# In[11]:


#Viewing arrays and matrices
un=np.unique(random)
un


# In[12]:


un[1]


# In[14]:


a=np.random.randint(10,size=(2,3,4,5))
a


# In[17]:


a.shape, a.ndim


# In[22]:


#Manipulating & comparing Arrays
#1. Arithemetic 

a1=np.array([1,2,3])
a1


# In[23]:


ones=np.ones(3)
ones


# In[24]:


a1+ones


# In[25]:


a1-ones


# In[27]:


a1*ones


# In[30]:


a2=np.array([[1,2,3.3],[4,5,6.5]])
a2


# In[39]:


a2*a1  #Cartesian product
a1//a1
np.square(a2)


# In[40]:


#How to reshape a2 to be compatible with all
#Search: "How to reshape numpy array"
listy_list=[1,2,3,443,34,23,6,657786,]


# In[41]:


#Aggregation-performin same operation on a number of things
type(listy_list)


# In[42]:


sum(listy_list)


# In[44]:


np.sum(listy_list)


# In[45]:


#Create a masssive numpy array 
massive_array=np.random.random(10000)
massive_array.size


# In[47]:


massive_array[:10]


# In[48]:


get_ipython().run_line_magic('timeit', 'sum(massive_array) #Python sum')
get_ipython().run_line_magic('timeit', 'np.sum(massive_array) #Numpy sum- This is more fast')


# In[49]:


a2


# In[51]:


np.mean(a2)


# In[55]:


np.std(a2)


# In[56]:


np.var(a2)


# In[57]:


np.sqrt(np.var(a2))


# In[58]:


#Demo of std and Var
high_var_array=np.array([1,100,200,300,4000,5000])
low_var_array=np.array([2,4,6,8,10])


# In[59]:


np.var(high_var_array),np.var(low_var_array)


# In[61]:


np.std(high_var_array),np.std(low_var_array)


# In[62]:


np.mean(high_var_array),np.mean(low_var_array)


# In[64]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.hist(high_var_array)
plt.show()


# In[65]:


plt.hist(low_var_array)
plt.show()


# In[96]:


##Reshaping and transposing 
a2


# In[98]:


a2.reshape(2,3,1).shape


# In[99]:


a2.T #Transpose switches the axis


# In[ ]:





# In[92]:


#Numpy dot product
np.random.seed(0)

mat1=np.random.randint(10, size=(5,3))
mat2=np.random.randint(10, size=(5,3))
mat1


# In[93]:


mat1.shape, mat2.shape


# In[94]:


#Element-wise multiplication (Hadamard product)
mat1*mat2


# In[95]:


#Dot product
mat1.shape
mat1.T.shape


# In[86]:


mat1.shape, mat2.T.shape


# In[88]:


mat3=np.dot(mat1,mat2.T)
mat3


# ###DOT Product example(Nut butter sale)

# In[101]:


np.random.seed(0)
#Number of jars sold
sales_amount=np.random.randint(20, size=(5,3))
sales_amount


# In[106]:


#Create weekly sales DataFrame 
weekly_sales = pd.DataFrame(sales_amount,
                            index=["Mon","Tue","Wed","Thurs","Fri"],
                            columns=["Almond butter","Peanut Butter","Cashew butter"])
weekly_sales


# In[107]:


#Create prices array
prices = np.array([10,8,12])
prices


# In[109]:


prices.shape


# In[110]:


#Create butter_prices DataFrame
butter_prices=pd.DataFrame(prices.reshape(1,3),
                          index=["Price"],
                          columns=["Almond butter","Peanut Butter","Cashew butter"])
butter_prices


# In[115]:


total_sales=prices.dot(sales_amount.T)
total_sales, weekly_sales.shape


# In[ ]:


weekly_sales


# In[2]:


#Shapes arn't aligned, lets transpose
total_sales=prices.dot(weekly_sales.T)
total_sales

