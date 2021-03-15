#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[2]:


COMPANY_LOCATION = "data/company_data.json"
INVESTOR_LOCATION = "data/investors_data.json"
JOBS_LOCATION = "data/jobs_data.json"


# # Load Data

# In[3]:


def load(filename):
    with open(filename, 'r') as file:
        datadict = json.load(file)
        return pd.DataFrame(datadict)

def load_companies():
    return load(COMPANY_LOCATION).transpose()

def load_investors():
    return load(INVESTOR_LOCATION)

def load_jobs():
    with open(JOBS_LOCATION, 'r') as file:
        return json.load(file)
    
#companies = load_companies()
#investors = load_investors()
#jobs = load_jobs()


# # Combining datasets
# 
# Turning the job data into more managable summary statistics surrounding the job openings

# In[4]:


def get_all_job_categories(jobs):
    all_categories = set()
    for company in jobs.values():
        for job in company:
            for category in job['attributes']['categories']:
                all_categories.add(category)
    return all_categories

def get_jobs_per_category(jobs, categories):
    all_companies = dict()
    for companyname, company in jobs.items():
        all_companies[companyname] = {cat: 0 for cat in categories}
        total = len(company)
        all_companies[companyname]["Total"] = total
        for job in company:
            for category in job['attributes']['categories']:
                all_companies[companyname][category] += 1
    
    df = pd.DataFrame(all_companies).transpose()
    #moves Total column to the end
    columns = df.columns.tolist()
    columns.remove('Total')
    columns.append('Total')
    return df[columns]


# In[5]:


def get_full_combined_companies_dataframe(jobs, companies):
    categories = get_all_job_categories(jobs)
    jobdf = get_jobs_per_category(jobs, categories)
    jobdf.columns = jobdf.columns.map(lambda x: 'jobs_' + str(x))
    return pd.merge(companies, jobdf, left_index=True, right_index=True)


# In[6]:


#full_companies = get_full_combined_companies_dataframe(jobs, companies)


# # Drop irrelevant Columns
# 
# List numbered columns, and select numbers with possible relevance to prediction

# In[7]:


def list_all_columns(fulldata):
    for num, col in enumerate(list(fulldata.columns)):
        print(str(num) + ' -- ' + col)

#list_all_columns(full_companies)


# In[8]:


DEFAULT_SELECTED_COLUMNS = [16,17,18,19,20,21,25,26,27,28,29,31,38,
                            39,44,45,63,64,74,75,77,78,81,83] + list(range(110,136))


# In[9]:


def drop_irrelevant_columns(fulldata, selected_columns=None):
    if selected_columns == None:
        selected_columns = DEFAULT_SELECTED_COLUMNS
    return fulldata.iloc[:,selected_columns]


# In[10]:


#relevant_data = drop_irrelevant_columns(full_companies)


# # Preprocess complex features
# 
# Many of the features contain lists, of categories, tags, founders. These can be transformed to one-hot encodings, or simply choosing the last value, depending on the feature.

# In[11]:


#list_all_columns(relevant_data)


# In[12]:


DEFAULT_LIST_FEATURES = [6,7,9,10,22,23]


# In[13]:


def transform_list_features_to_onehot(dataset, featurelist=None):
    output_df = dataset.copy()
    if featurelist == None:
        featurelist = DEFAULT_LIST_FEATURES
    columns = dataset.columns[featurelist]
    for column in columns:
        if type(dataset[column].iloc[0]) == list:
            feature_df = transform_list_feature_to_onehot(dataset[column])
            output_df = output_df.drop(column, axis=1)
            output_df = pd.merge(output_df, feature_df,left_index=True, right_index=True)
    return output_df

def transform_list_feature_to_onehot(feature):
    all_values = {value for entry in feature for value in entry}
    output_data = dict()
    for index, entry in feature.items():
        outdict = {value: 0 for value in all_values}
        for value in entry:
            outdict[value] += 1
        output_data[index] = outdict
    output_df = pd.DataFrame(output_data).transpose()
    output_df.columns = output_df.columns.map(lambda x: str(feature.name) + '_' + str(x))
    return output_df               


# In[14]:


def select_last_item(l):
    if len(l) > 0:
        return l[-1]
    else:
        return np.nan

def list_to_latest_and_count(dataset, featurelist=None):
    if featurelist==None:
        featurelist = [col for col in dataset.columns if type(dataset[col][0]) == list]
    for feature in featurelist:
        new_feature_df = pd.DataFrame({feature+'_last' : dataset[feature].apply(lambda x: select_last_item(x)),
                          feature+'_count' : dataset[feature].apply(lambda x: len(x))})
        dataset = dataset.drop(feature, axis=1)
        dataset = pd.merge(dataset, new_feature_df,left_index=True, right_index=True)
    return dataset

def transform_all_list_features(dataset, onehotfeatures=None, lastvaluefeatures=None):
    output = transform_list_features_to_onehot(dataset, onehotfeatures)
    output = list_to_latest_and_count(output, lastvaluefeatures)
    return output


# In[15]:


#transformed_data = transform_all_list_features(relevant_data)


# ## Fixing annoying datatypes
# 
# Things like dates, number ranges etc.

# In[16]:


DEFAULT_RANGE_FEATURES = ['kpi_valuation_eur']
DEFAULT_DATE_FEATURES = ['last_funding_round_date', 'first_funding_round', 'launch_date', 'kpi_valuation_date', 
                         'employees_date_last']


# In[17]:


def replace_na_with_nan(datapoint):
    if datapoint == 'n/a':
        return np.nan
    elif datapoint == None:
        return np.nan
    else:
        return datapoint

def replace_all_na_with_nan(dataset):
    for column in dataset.columns:
        dataset.loc[:,column] = dataset[column].apply(lambda x: replace_na_with_nan(x))
    return dataset


# In[20]:


def range_to_average(data_range):
    if type(data_range) == str:
        divided = data_range.strip(" ' ").split('-')
        divided[0] = int(divided[0])
        divided[1] = int(divided[1])
        return sum(divided)/len(divided)
    else:
        return data_range
    
def ranges_to_average(dataset, range_features = None):
    if range_features == None:
        range_features = DEFAULT_RANGE_FEATURES
    for feature in range_features:
        dataset.loc[:,feature] = dataset[feature].apply(lambda x: range_to_average(x))
    return dataset

def date_to_int(date):
    for index, char in enumerate(str(date)):
        if char == '/':
            return str(date)[index+1:]
        elif len(str(date)) > index+1:
            if (char == '1' and str(date)[index+1] == '9') or (char == '2' and str(date)[index+1] == '0'):
                return str(date)[index:index+4]

def dates_to_int(dataset, date_features = None):
    if date_features == None:
        date_features = DEFAULT_DATE_FEATURES
    for feature in date_features:
        dataset.loc[:,feature] = dataset[feature].apply( lambda x: date_to_int(x))
    return dataset

def all_to_numeric(dataset):
    for feature in dataset.columns:
        try:
            dataset.loc[:,feature] = pd.to_numeric(dataset[feature])
        except:
            pass
    return dataset        
    
def fix_all_data_types(dataset, range_features = None, date_features = None):
    output = ranges_to_average(dataset, range_features)
    output = dates_to_int(output, date_features)
    output = replace_all_na_with_nan(output)
    output = all_to_numeric(output)
    output = output.infer_objects()
    return output


# In[50]:


#transformed_fixed_data = fix_all_data_types(transformed_data)


# In[60]:


DEFAULT_IRRELEVANT_COLUMNS = ['profit_values_report_count', 'ebitda_values_report_count', 'employees_date_count',
                              'employees_values_count', 'each_funding_rounds_amount_count', 
                              'each_funding_rounds_date_count', 'each_funding_rounds_amount_last', 
                              'each_funding_rounds_date_last', 'each_funding_rounds_type_last', 
                              'profit_values_report_last', 'ebitda_values_report_last',
                              'each_funding_rounds_type_count',]


# In[61]:


def drop_final_irrelevant_columns(dataset, columns=None):
    if columns == None:
        columns = DEFAULT_IRRELEVANT_COLUMNS
    return dataset.drop(columns, axis=1)

def onehot_final_objects(dataset):
    return pd.get_dummies(dataset)


# In[62]:

def fix_spoilers(datapoint):
    if datapoint['fundings_total'] == 0:
        datapoint.loc['first_funding_round'] = np.nan
    return datapoint

def fix_spoiler_features(dataset):
    dataset.loc[:,'total_funding_report'] = dataset['total_funding_report']-dataset['last_funding']
    dataset.loc[:,'fundings_total'] = dataset['fundings_total']-1
    dataset = dataset.apply(lambda x: fix_spoilers(x), axis=1)
    return dataset

#test = drop_final_irrelevant_columns(transformed_fixed_data)


# In[63]:


#test = onehot_final_objects(test)


# In[65]:


def get_final_dataset():
    companies = load_companies()
    jobs = load_jobs()
    full_companies = get_full_combined_companies_dataframe(jobs, companies)
    relevant_data = drop_irrelevant_columns(full_companies)
    transformed_data = transform_all_list_features(relevant_data)
    transformed_fixed_data = fix_all_data_types(transformed_data)
    transformed_relevant = drop_final_irrelevant_columns(transformed_fixed_data)
    fully_transformed = onehot_final_objects(transformed_relevant)
    final_data = fix_spoiler_features(fully_transformed)
    return final_data


# In[66]:


#final_data = get_final_dataset()


# In[67]:

def split_and_save(dataset, test_size=0.2, filename="dataset"):
    train, test = train_test_split(dataset, test_size=test_size, random_state=42)
    with open(filename+'_train','wb') as file:
        pickle.dump(train, file)
    with open(filename+'_test', 'wb') as file:
        pickle.dump(test, file)



# In[ ]:




