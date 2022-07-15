#!/usr/bin/env python
# coding: utf-8

# #### Importing packages needed for preparing the data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# #### Reading the data

# In[2]:


onlinec2c=pd.read_csv('onlinec2c.csv') 


# In[3]:


onlinec2c.head()


# #### Data Understanding

# In[4]:


onlinec2c.info()


# In[5]:


onlinec2c.describe()


# #### Data Cleaning

# From the descriptive analysis, we see that there are many variables which are not balanced in proportion and therefore they possibly have outliers. Next step in data cleaning is to check outliers for each suspicious variables using boxplot 

# In[6]:


plt.boxplot(onlinec2c['daysSinceLastLogin'])


# In[7]:


onlinec2c2=onlinec2c.loc[onlinec2c['daysSinceLastLogin']<366]
onlinec2c2.info()


# In[8]:


onlinec2c2.describe()


# In[9]:


plt.boxplot(onlinec2c2['socialNbFollowers'])


# In[10]:


onlinec2c3=onlinec2c2.loc[onlinec2c2['socialNbFollowers']<201]
onlinec2c3.info()


# In[11]:


onlinec2c3.describe()


# In[12]:


plt.boxplot(onlinec2c3['productsListed'])


# In[13]:


onlinec2c4=onlinec2c3.loc[onlinec2c3['productsListed']<71]
onlinec2c4.info()


# In[14]:


onlinec2c4.describe()


# In[15]:


onlinec2c5=onlinec2c4.loc[onlinec2c4['socialProductsLiked']<2001]
onlinec2c5.info()


# In[16]:


onlinec2c5.describe()


# In[17]:


plt.boxplot(onlinec2c5['productsWished'])


# In[18]:


onlinec2c6=onlinec2c5.loc[onlinec2c5['productsWished']<2]
onlinec2c6.info()


# In[19]:


onlinec2c6.describe()


# In[20]:


plt.boxplot(onlinec2c6['productsSold'])


# In[21]:


onlinec2c7=onlinec2c6.loc[onlinec2c6['productsSold']<35]
onlinec2c7.info()


# In[22]:


onlinec2c7.describe()


# In[23]:


plt.boxplot(onlinec2c7['productsPassRate'])


# In[24]:


onlinec2c8=onlinec2c7.loc[onlinec2c7['productsPassRate']<41]
onlinec2c8.info()


# In[25]:


onlinec2c8.describe()


# In[26]:


plt.boxplot(onlinec2c8['productsBought'])


# In[27]:


onlinec2c9=onlinec2c8.loc[onlinec2c8['productsBought']<20]
onlinec2c9.info()


# In[28]:


plt.boxplot(onlinec2c9['seniority'])


# In[29]:


np.array(list(onlinec2c9.columns),dtype=object)


# As a result of the data cleaning process, we now have the variables as above 

# #### Data Preparation

# #### Label Encoding

# The purpose of this label encoding process is to divide the datapoints of the desired variables into new proportion

# In[30]:


def value_counts(onlinec2c9, col, style=True):
    table = onlinec2c9[col].value_counts().rename_axis('Value').reset_index(name='Count')
    table['Percentage'] = table['Count'] / table['Count'].sum(axis=0)
    
    if style:
        table = table.style.format({'Count': '{:,}', 'Percentage': '{:.2%}'}).hide_index()
        
    return table


# In[31]:


value_counts(onlinec2c9, 'country')


# In[32]:


CCT=[] 
for i,j in enumerate (onlinec2c9['country']):
    if j=='France':
        CCT.append(1)
    elif j=='Etats-Unis':
        CCT.append(2) 
    elif j=='Royaume-Uni':
        CCT.append(3)
    elif j=='Italie':
        CCT.append(4)
    elif j=='Allemagne':
        CCT.append(5)
    elif j=='Espagne':
        CCT.append(6)
    elif j=='Australie':
        CCT.append(7)
    elif j=='Danemark':
        CCT.append(8)
    elif j=='Suède':
        CCT.append(9)
    elif j=='Belgique':
        CCT.append(10)
    elif j=='Canada':
        CCT.append(11)
    elif j=='Pays-Bas':
        CCT.append(12)
    elif j=='Suisse':
        CCT.append(13)
    elif j=='Finlande':
        CCT.append(14)
    elif j=='Hong Kong':
        CCT.append(15)
    else:
        CCT.append(0)
onlinec2c9['Common_Country_Type']=CCT
onlinec2c9.head()


# In[33]:


value_counts(onlinec2c9, 'language')


# In[34]:


CLT=[] 
for i,j in enumerate (onlinec2c9['language']):
    if j=='en':
        CLT.append(1)
    elif j=='fr':
        CLT.append(2) 
    elif j=='it':
        CLT.append(3)
    elif j=='de':
        CLT.append(4)
    else:
        CLT.append(0)
onlinec2c9['Common_Language_Type']=CLT
onlinec2c9.head()


# In[35]:


value_counts(onlinec2c9, 'civilityTitle')


# In[36]:


value_counts(onlinec2c9, 'gender')


# In[37]:


value_counts(onlinec2c9, 'civilityGenderId')


# In[38]:


CGT=[] 
for i,j in enumerate (onlinec2c9['gender']):
    if j=='F':
        CGT.append(1)
    else:
        CGT.append(0)
onlinec2c9['Common_Gender_Type']=CGT
onlinec2c9.head()


# In[39]:


CAT=[] 
for i,j in enumerate (onlinec2c9['hasAnyApp']):
    if j==True:
        CAT.append(1)
    else:
        CAT.append(0)
onlinec2c9['Common_AnyApp_Type']=CAT
onlinec2c9.head()


# In[40]:


CAAT=[] 
for i,j in enumerate (onlinec2c9['hasAndroidApp']):
    if j==True:
        CAAT.append(1)
    else:
        CAAT.append(0)
onlinec2c9['Common_AndroidApp_Type']=CAAT
onlinec2c9.head()


# In[41]:


CIAT=[] 
for i,j in enumerate (onlinec2c9['hasIosApp']):
    if j==True:
        CIAT.append(1)
    else:
        CIAT.append(0)
onlinec2c9['Common_IosApp_Type']=CIAT
onlinec2c9.head()


# In[42]:


CPPT=[] 
for i,j in enumerate (onlinec2c9['hasProfilePicture']):
    if j==True:
        CPPT.append(1)
    else:
        CPPT.append(0)
onlinec2c9['Common_ProfilePicture_Type']=CPPT
onlinec2c9.head()


# After going through label encoding process, we would want to take the integer variables only

# In[43]:


c2c=onlinec2c9.drop(['identifierHash','countryCode','gender','type','civilityTitle','country','language','hasAnyApp', 'hasAndroidApp', 'hasIosApp', 'hasProfilePicture','seniorityAsYears','seniorityAsMonths'],axis=1)
c2c.head()


# In[44]:


c2c=c2c.reset_index(drop=True)
c2c.head()


# From the VIF below, we could say that there is multicollinearity on some variables such civilityGenderId, socialNbFollows, and other variables which has VIF greater than 10

# In[45]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["variables"] = c2c.columns
vif["VIF"] = [variance_inflation_factor(c2c.values, i) for i in range(c2c.shape[1])]

print(vif)


# In[46]:


c2c=c2c.drop(['socialNbFollows','civilityGenderId','seniority','Common_Gender_Type','Common_AnyApp_Type','Common_AndroidApp_Type','Common_IosApp_Type','Common_ProfilePicture_Type'],axis=1)
c2c.head()


# As a result, we now only have ten variables as we can see in the table above

# #### Data Analysis

# #### Importing required packages for data analysis process

# In[47]:


pip install factor_analyzer


# #### Kaiser Meyer Olkin (KMO TEST)

# KMO Test measures the proportion of variance that might be a common variance among the variables. Larger proportions are expected as it represents more correlation is present among the variables thereby giving way for the application of dimensionality reduction techniques such as Factor Analysis. KMO score is always between 0 to 1 and values more than 0.6 are good. We can also say it as a measure of how suited our data is for factor analysis.

# In[48]:


from factor_analyzer import calculate_kmo
kmo_vars,kmo_model = calculate_kmo(c2c)
print(kmo_model)


# In[49]:


from sklearn.preprocessing import StandardScaler


# In[50]:


from factor_analyzer import FactorAnalyzer


# We would want to do a scaling for our data to prepare it for the next analysis process

# In[51]:


scaler =  StandardScaler()
dataframe = scaler.fit_transform(c2c)
dataframe = pd.DataFrame(data=dataframe,columns=c2c.columns)
dataframe.head(10)


# In[52]:


fa = FactorAnalyzer(rotation = None,impute = "drop",n_factors=dataframe.shape[1])


# In[53]:


fa.fit(dataframe)


# In[54]:


ev,_ = fa.get_eigenvalues()


# #### Bartlett

# In[55]:


from factor_analyzer import calculate_bartlett_sphericity
chi2,p = calculate_bartlett_sphericity(dataframe)
print("Chi squared value : ",chi2)
print("p value : ",p)


# Since the p test statistic is less than 0.05, we can conclude that correlation is present among the variables which is a green signal to apply factor analysis.

# #### Determining number of factors

# In[56]:


plt.scatter(range(1,dataframe.shape[1]+1),ev)
plt.plot(range(1,dataframe.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigen Value')
plt.grid()


# The number of factors can be decided on the basis of the amount of common variance the factors explain. In general, we will plot the factors and their eigenvalues.
# 
# Eigenvalues are nothing but the amount of variance the factor explains. We will select the number of factors whose eigenvalues are greater than 1.
# 
# The eigenvalues function will return the original eigenvalues and the common factor eigenvalues. Now, we are going to consider only the original eigenvalues. From the graph, we can see that the eigenvalues drop below 1 from the 4th factor. So, the optimal number of factors is 4.

# #### Interpreting the Factors

# In[57]:


fa = FactorAnalyzer(n_factors=4,rotation='varimax')
fa.fit(c2c)


# In[58]:


with np.printoptions(suppress=True,precision=4):
    print(pd.DataFrame(fa.get_eigenvalues()[0],columns=['EigenValues']))


# #### Loading score

# In[59]:


with np.printoptions(suppress=True,precision=4):
    print(pd.DataFrame(fa.loadings_,index=dataframe.columns))


# Loadings indicate how much a factor explains a variable. The loading score will range from -1 to 1.Values close to -1 or 1 indicate that the factor has an influence on these variables. Values close to 0 indicates that the factor has a lower influencer on the variable.
# 
# For example, in Factor 0, we can see that the features ‘productsSold’, 'socialNbFollowers' and ‘productsPassRate’ have high loadings than other variables. From this, we can see that Factor 0, explains the common variance in categories which are reserved i.e. the variance among the users' interaction which are included in 'productsSold', 'productPassRate', and 'socialNbFollowers'.

# #### Variance

# In[60]:


with np.printoptions(suppress=True,precision=4):
    print(pd.DataFrame(fa.get_factor_variance(),index=['Variance','Proportional Var','Cumulative Var']))


# The first row represents the variance explained by each factor. Proportional variance is the variance explained by a factor out of the total variance. Cumulative variance is nothing but the cumulative sum of proportional variances of each factor. In our case, the 4 factors together are able to explain 34.1% of the total variance.
# 
# In unrotated cases, the variances would be equal to the eigenvalues. Rotation changes the distribution of proportional variance but the cumulative variance will remain the same. Oblique rotations allow correlation between the factors while the orthogonal rotations keep the factors uncorrelated.

# **Communalities**

# Communality is the proportion of each variable’s variance that can be explained by the factors. Rotations don’t have any influence over the communality of the variables

# In[61]:


with np.printoptions(precision=4,suppress=True):
    print(pd.DataFrame(fa.get_communalities(),index=dataframe.columns,columns=['Communalities']))


# The proportion of each variable’s variance that is explained by the factors can be inferred from the above. For example, we could consider the variable ‘productsSold’ about 100.1% of its variance is explained by all the factors together

# #### Confirming the relation of each variables using correlation table

# In[62]:


c2c.corr()


# #### Hirarchieral Clustering

# In[63]:


from sklearn.preprocessing import normalize
data_scaled = normalize(c2c)
data_scaled = pd.DataFrame(data_scaled, columns=c2c.columns)
data_scaled.head()


# In[64]:


import scipy.cluster.hierarchy as shc


# In[65]:


data_scaled.info()


# In[66]:


plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))


# The x-axis contains the samples and y-axis represents the distance between these samples. The vertical line with maximum distance is the blue line and hence we can decide a threshold of 15 and cut the dendrogram:

# In[67]:


plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
plt.axhline(y=15, color='r', linestyle='--')


# We have two clusters as this line cuts the dendrogram at two points. Let’s now apply hierarchical clustering for 2 clusters:

# In[68]:


from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
cluster.fit_predict(data_scaled)


# In[69]:


plt.figure(figsize=(10, 7))  
plt.scatter(data_scaled['socialNbFollowers'], data_scaled['socialProductsLiked'], c=cluster.labels_) 


# In[70]:


plt.figure(figsize=(10, 7))  
plt.scatter(data_scaled['socialProductsLiked'], data_scaled['Common_Country_Type'], c=cluster.labels_) 


# In[71]:


plt.figure(figsize=(10, 7))  
plt.scatter(data_scaled['productsBought'], data_scaled['socialProductsLiked'], c=cluster.labels_) 


# In[72]:


plt.figure(figsize=(10, 7))  
plt.scatter(data_scaled['daysSinceLastLogin'], data_scaled['socialProductsLiked'], c=cluster.labels_) 


# Above are some combination of variables to divide the datapoints into to clusters. We see that almost all scatterplots give a quite distinguished area division of clusters

# In[73]:


data_scaled.corr()


# Seeing from the correlation table above and the scatterplot visualization, the better division is generated from two variables which have low correlation

# In[74]:


c2c.describe()


# In[75]:


group=cluster.labels_
c2c['cluster']=group
c2c.head()


# In[76]:


ccluster0=c2c.loc[c2c['cluster']==0]
ccluster0.describe()


# In[77]:


ccluster0.loc[(ccluster0['socialProductsLiked']>0)].count()


# In[78]:


ccluster0.loc[(ccluster0['productsWished']>0)].count()


# In[79]:


ccluster0.loc[(ccluster0['productsBought']>0)].count()


# In[80]:


ccluster0.loc[(ccluster0['productsListed']>0)].count()


# In[81]:


ccluster0.loc[(ccluster0['productsSold']>0)].count()


# In[82]:


ccluster1=c2c.loc[c2c['cluster']==1]
ccluster1.describe()


# In[83]:


ccluster1.loc[(ccluster1['socialProductsLiked']>0)].count()


# In[84]:


ccluster1.loc[(ccluster1['productsWished']>0)].count()


# In[85]:


ccluster1.loc[(ccluster1['productsBought']>0)].count()


# In[86]:


ccluster1.loc[(ccluster1['productsListed']>0)].count()


# In[87]:


ccluster1.loc[(ccluster1['productsSold']>0)].count()


# The cluster 1 suggest that it is the active user group and cluster 0 is the passive user group
