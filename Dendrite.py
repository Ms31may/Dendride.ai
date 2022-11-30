#!/usr/bin/env python
# coding: utf-8

# # Dendrite.ai Screening Test
# 

# In[9]:


import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor 
LabelEncoder = LabelEncoder()
from sklearn.feature_selection import RFE,SelectFromModel
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
grid =RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import sklearn
from sklearn.metrics import r2_score, classification_report


# In[10]:


with open("C:\\Users\\mohit\\Downloads\\Screening Test - DS\\Screening Test\\algoparams_from_ui.json", 'r')as file:
    text =json.load(file)

x = pd.DataFrame(text)
dataset = x['design_state_data']['session_info']['dataset']


# In[11]:


df = pd.read_csv(dataset)


# In[ ]:





# ## 1)	Read the target and type of regression to be run

# In[12]:


target = x['design_state_data']['target']['target']
prediction_type = x['design_state_data']['target']['type']


# ## 2) Read the features(which are column names in the csv) and figure out what missing imputation needs to be applied and apply that to the columns loaded in a dataframe 

# In[13]:


feature_handling  = x['design_state_data']['feature_handling']
columns= []
for k in feature_handling:
    if (feature_handling[k]['is_selected'] == True) & (k!=target) :
        columns.append(k)
        if feature_handling[k]['feature_details'].get('impute_with') != None:
            if feature_handling[k]['feature_details']['impute_with'] == 'Average of values':
                df[k] = df[k].fillna((df[k].mean()))
            elif feature_handling[k]['feature_details']['impute_with'] == "custom":
                impute_value = feature_handling[k]['feature_details']['impute_value']
                df[k] = df[k].fillna(impute_value)
        if (feature_handling[k]['feature_variable_type'] == "text"):
            df[k] =LabelEncoder.fit_transform(df[k])
            
        
    


# In[ ]:





# ## 3)	Compute feature reduction based on input. See the screenshot below where there can be No Reduction, Corr with Target, Tree-based, PCA. Please make sure you write code so that all options can work. If we rerun your code with a different json it should work if we switch No Reduction to say PCA. 

# In[14]:


feature_reduction = x['design_state_data']['feature_reduction']
tr = df[columns]
te = df[target]
if (feature_reduction['feature_reduction_method'] == 'Tree-based'):
    # With feature selection check auuracy with Random Forest
    # The following example shows how to retrieve the 7 most informative features
    model_tree = RandomForestRegressor(n_estimators = int(feature_reduction['num_of_trees']), max_depth =int(feature_reduction['depth_of_trees']))

    # use RFE to eleminate the less importance features
    sel_rfe_tree = RFE(estimator=model_tree, n_features_to_select=int(feature_reduction['num_of_features_to_keep']))
    X_train_rfe_tree = sel_rfe_tree.fit_transform(tr, te)
    selected_cols = [column for column in tr.columns if column in tr.columns[sel_rfe_tree.get_support()]]
else:
    selected_cols = columns
    




# In[15]:


algorithm = feature_reduction = x['design_state_data']['algorithms']
hyperparameters = feature_reduction = x['design_state_data']['hyperparameters']


# In[ ]:





# In[19]:


tr = df[selected_cols]
te = df[target]
X_train, X_test,y_train, y_test = train_test_split(tr,te,test_size=0.25, shuffle=True)

if x['design_state_data']['target']['type'] == 'regression':
    k ="RandomForestRegressor"
    model = RandomForestRegressor()
    mintrees = int(algorithm[k]['min_trees'])
    maxtrees = int(algorithm[k]['max_trees'])
    min_depth = int(algorithm[k]['min_depth'])
    max_depth = int(algorithm[k]['max_depth'])
    minleafsamplei = int(algorithm[k]['min_samples_per_leaf_min_value'])
    minleafsamplej = int(algorithm[k]['min_samples_per_leaf_max_value'])

    forest_param = [{'n_estimators':list(range(mintrees, (maxtrees+1))),'max_depth': list(range(min_depth, (max_depth+1))), 'min_samples_leaf':list(range(minleafsamplei, (minleafsamplej+1)))}]



else :
    k= 'DecisionTreeClassifier'
    model = sklearn.ensemble.RandomForestClassifier()
    mintrees = int(algorithm[k]['min_trees'])
    maxtrees = int(algorithm[k]['max_trees'])
    min_depth = int(algorithm[k]['min_depth'])
    max_depth = int(algorithm[k]['max_depth'])
    minleafsamplei = int(algorithm[k]['min_samples_per_leaf_min_value'])
    minleafsamplej = int(algorithm[k]['min_samples_per_leaf_max_value'])

    forest_param = [{'n_estimators':list(range(mintrees, (maxtrees+1))),'max_depth': list(range(min_depth, (max_depth+1))), 'min_samples_leaf':list(range(minleafsamplei, (minleafsamplej+1)))}]



    

        


# ## Run the fit and predict on each model  keep in mind that you need to do hyper parameter tuning i.e. use GridSearchCV

# In[20]:


if (hyperparameters['stratergy'] == 'Grid Search'):
    rf_random = grid(estimator = model, param_distributions = forest_param, n_iter = int(hyperparameters['max_iterations']), cv = int(hyperparameters['num_of_folds']), verbose=2, random_state=int(hyperparameters['random_state']), n_jobs = int(hyperparameters['parallelism']), scoring='neg_mean_squared_error')
    
    
rf_random.fit(X_train, y_train)
rf_random.best_estimator_


# ## Log to the console the standard model metrics that apply

# In[23]:


final_model = rf_random.best_estimator_
# Predicting test set results
final_pred = final_model.predict(X_test)
if x['design_state_data']['target']['type'] == 'regression':
    print("R2 Value of model is",r2_score(y_test, final_pred))
else:
    print(classification_report(y_test, final_pred))


# In[ ]:





# In[ ]:





# In[ ]:




