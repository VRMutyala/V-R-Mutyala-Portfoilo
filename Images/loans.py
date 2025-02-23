#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from feature_engine.outliers import Winsorizer
from sklearn.impute import SimpleImputer
import scipy.stats as stats
import pylab
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import re 


# In[15]:


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load dataset (Ensure correct file format: .csv or .xlsx)
file_path = r"C:\Users\Harin\Downloads\loans (1).csv"  # Use raw string (r"...")

# If the file is CSV, use pd.read_csv
data = pd.read_csv(file_path, encoding="ISO-8859-1", low_memory=False)

logging.info("Dataset loaded successfully.")


# In[16]:


data.sample(10)


# In[17]:


data.info()


# In[18]:


#All columns having ? in place of e (example :'time')
# Function to clean column names
def clean_column_name(col):
    return re.sub(r"[?]", "e", col)  # Replace '?' with 'e'

# Apply function to all column names
data.columns = [clean_column_name(col) for col in data.columns]

print(data.columns)  # Verify fixed column names


# In[21]:


data.dtypes


# In[20]:


# columns having j0int in place of joint 
data.rename(columns=lambda x: x.replace("j0int", "joint"), inplace=True)


# In[22]:


data.drop(columns=['Title','num_accounts_120d_past_due','num_accounts_30d_past_due','Time.1','Time','issue_month',
          'Status ','annual_income_joint'],inplace=True)


# In[23]:


# Convert Date column to datetime format if present
if 'issue_month.1' in data.columns:
    data['issue_month.1'] = pd.to_datetime(data['issue_month.1'], errors='coerce')


# In[24]:


#merging grade and subgarde columns  
data['merged_grade'] = data['grade'] + '-' + data['sub_grade']
data.drop(columns=['grade', 'sub_grade'], inplace=True)


# In[25]:


data.head()


# In[26]:


#in my data rows also having missing 
data = data.applymap(lambda x: x.replace("?", "e") if isinstance(x, str) else x)


# In[27]:


#correction of names by first letter capitalize 
data = data.applymap(lambda x: x.capitalize() if isinstance(x, str) else x)


# In[28]:


# Handle duplicate rows
duplicates = data.duplicated().sum()
logging.info(f"Duplicate Rows: {duplicates}")
data.drop_duplicates(inplace=True)


# In[29]:


# Summary statistics
pd.DataFrame({'count': data.shape[0], 'unique': data.nunique()})


# In[30]:


demographic_columns = [
    'emp_title', 'emp_length', 'state', 'homeownership', 'annual_income', 
    'verified_income', 'verification_income_joint', 'application_type'
]

# Extract only the required columns
demographic_df = data[demographic_columns]

# Save to CSV
demographic_df.to_csv('demographic_columns.csv', index=False)

print("CSV file saved successfully!")


# In[31]:


# Fix values inside 'application_type' where 'J0int' is incorrectly written
data['application_type'] = data['application_type'].replace("J0int", "joint")


# In[72]:


print(type(demographic_columns))


# In[32]:


# For categorical columns, fill missing with the mode (most frequent value)
for col in data[demographic_columns].select_dtypes(include='object').columns:
    data[col].fillna(data[col].mode()[0], inplace=True)


# In[33]:


# For numerical columns, fill missing with the median (for 'annual_income', 'emp_length')
for col in data[demographic_columns].select_dtypes(include='number').columns:
    data[col].fillna(data[col].median(), inplace=True)


# In[34]:


#2. Remove Duplicates
data.drop_duplicates(subset=demographic_columns, inplace=True)

#  Convert Data Types if necessary
# Ensure numerical columns are of numeric type (if they aren't already)
data['annual_income'] = pd.to_numeric(data['annual_income'], errors='coerce')
data['emp_length'] = pd.to_numeric(data['emp_length'], errors='coerce')


# In[35]:


# Calculate the median of the 'income' column (ignoring NaN)
median_income = data['annual_income'].median()

# Replace NaN (missing) values in 'income' with the median value
data['annual_income'].fillna(median_income, inplace=True)


# In[36]:


print(data.info())


# In[ ]:





# In[37]:


# grouping debit and credit columns 
debt_and_credit_columns = [
    'debt_to_income', 'debt_to_income_joint', 'total_credit_lines', 'open_credit_lines', 
    'total_credit_limit', 'total_credit_utilized', 'num_collections_last_12m', 'num_historical_failed_to_pay', 
    'num_satisfactory_accounts','num_active_debit_accounts',
    'total_debit_limit', 'num_total_cc_accounts', 'num_open_cc_accounts', 'num_cc_carrying_balance', 'num_mort_accounts', 
    'tax_liens', 'public_record_bankrupt'
]

# Extract only the required columns
debt_and_credit_columns = data[debt_and_credit_columns]

# Save to CSV
debt_and_credit_columns.to_csv('debt_and_credit_columns.csv', index=False)

print("CSV file saved successfully!")


# In[38]:


debt_and_credit_columns.info()


# In[39]:


# Remove Duplicates
data.drop_duplicates(subset=debt_and_credit_columns, inplace=True)


# In[40]:


# List of columns with missing values (replace with your actual column names)
columns_to_fill = ['debt_to_income', 'debt_to_income_joint', 'open_credit_lines']

# Fill missing values with the median of each column
for col in columns_to_fill:
    data[col].fillna(data[col].median(), inplace=True)


# In[ ]:





# In[47]:


# credit history and enquries 
credit_history_and_inquiries_columns = [
    'delinq_2y', 'months_since_last_delinq', 'earliest_credit_line', 'inquiries_last_12m', 
    'months_since_last_credit_inquiry', 'months_since_90d_late', 'current_accounts_delinq', 
    'total_collection_amount_ever', 'current_installment_accounts', 'accounts_opened_24m'
]

# Extract only the required columns
credit_history_and_inquiries_columns = data[credit_history_and_inquiries_columns]

# Save to CSV
credit_history_and_inquiries_columns.to_csv('credit_history_and_inquiries_columns.csv', index=False)

print("CSV file saved successfully!")


# In[48]:


credit_history_and_inquiries_columns.info()


# In[43]:


# Remove Duplicates
data.drop_duplicates(subset=credit_history_and_inquiries_columns, inplace=True)


# In[45]:


# List of columns with missing values (replace with your actual column names)
columns_to_fill = ['months_since_last_delinq', 'earliest_credit_line', 'months_since_last_credit_inquiry','months_since_90d_late']

# Fill missing values with the median of each column
for col in columns_to_fill:
    data[col].fillna(data[col].median(), inplace=True)


# In[ ]:





# In[ ]:





# In[86]:


#loan information columns 
loan_information_columns = [
    'loan_purpose', 'loan_amount', 'term', 'interest_rate', 'installment', 'issue_month.1', 
    'loan_status', 'initial_listing_status', 'disbursement_method', 'balance', 'paid_total', 
    'paid_principal', 'paid_interest', 'paid_late_fees'
]

# Extract only the required columns
loan_information_columns = data[loan_information_columns]

# Save to CSV
loan_information_columns.to_csv('loan_information_columns.csv', index=False)

print("CSV file saved successfully!")


# In[76]:


loan_information_columns.info()


# In[63]:


# Remove Duplicates
data.drop_duplicates(subset=loan_information_columns, inplace=True)


# In[75]:


loan_information_columns.sample(10)


# In[83]:


cols_to_round = ['paid_total', 'paid_principal', 'paid_interest', 'paid_late_fees', 'balance']

# Convert columns to numeric (handle errors if any)
loan_information_columns.loc[:, cols_to_round] = loan_information_columns.loc[:, cols_to_round].apply(pd.to_numeric, errors='coerce')

# Force rounding and convert back to float to remove extra precision
loan_information_columns.loc[:, cols_to_round] = loan_information_columns.loc[:, cols_to_round].round(2)

# Check again if any values have more than two decimal places
not_rounded = loan_information_columns[cols_to_round].apply(lambda x: (x * 100) % 1 != 0).any()
print("Are there any unrounded values? ", not_rounded.any())  # Should return False

# Print sample rows to verify rounding
print(loan_information_columns[cols_to_round].head())


# In[84]:


# Find unrounded rows
unrounded_rows = loan_information_columns[cols_to_round].apply(lambda x: (x * 100) % 1 != 0)
print(loan_information_columns[unrounded_rows.any(axis=1)])


# In[85]:


# Define columns to download
cols_to_round = ['paid_total', 'paid_principal', 'paid_interest', 'paid_late_fees', 'balance']

# Save to CSV
loan_information_columns[cols_to_round].to_csv("rounded_loan_data.csv", index=False)

print("File saved successfully as 'rounded_loan_data.csv'")


# In[87]:


#grade 
grade_columns = ['merged_grade']


# In[ ]:





# In[ ]:





# In[88]:


#percentage 
percentages_columns = ['account_never_delinq_percent']


# In[ ]:




