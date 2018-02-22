import pandas as pd
import numpy as np
from sklearn import *
import xgboost as xgb
import re

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

params={"objective":"binary:logistic",     
    "learning_rate":0.1,
    "subsample":0.8,
    "colsample_bytree": 0.8,
        'eval_metric':'auc',
    "max_depth":6,
    'silent':1,
    'nthread':3,
#     'num_class':2
        
       }  
params_lgb = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'application': 'binary',
    'metric':'auc',
    'num_leaves': 128,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'num_threads':3,
    'max_bin':500,
    'verbose': -1
}
def last_week(string):
    string =re.findall('\d+',string )
    if len(string)==0:
        return -1
    return int(string[0])
    
data=train.append(test)
data.reset_index(drop=1,inplace=1)

data.last_week_pay=data.last_week_pay.apply(lambda x:last_week(x) )

data.loc[data.term=='36 months','term']=36*4
data.loc[data.term=='60 months','term']=60*4

data.term.head()


data['week_left']=data.last_week_pay/data.term

data['same_bank']=0
data.loc[data.loan_amnt!=data.funded_amnt,'same_bank']=1


data['same_investor']=0
data.loc[data.loan_amnt!=data.funded_amnt_inv,'same_investor']=1

data['total_loan']=data.funded_amnt*(data.int_rate/100)*3+data.funded_amnt
data.loc[data.term==240,'total_loan']=data.funded_amnt*(data.int_rate/100)*5+data.funded_amnt

data['loan_left']=(data.total_loan*(data.last_week_pay/data.term)+data.total_rec_int)/data.total_loan


data['inc_loan_ratio']=data.funded_amnt/data.annual_inc


data['monthly_loan']=(data.dti/100)*(data.annual_inc/12)
data['loan_total']=(data.dti/100)*(data.annual_inc/12)*data.term

data['loan_to_income']=(data.monthly_loan*12)/data.annual_inc


data['interest_month']=data.total_rec_int/data.last_week_pay
data.interest_month.replace(np.inf,-1,inplace=1)


data['tot_rec_int_ratio']=data.total_rec_int/(data.funded_amnt*(data.int_rate/100)*3)


data.loc[data.term==240,'tot_rec_int_ratio']=data.total_rec_int/(data.funded_amnt*(data.int_rate/100)*5)


data['left_interest']=1-data.tot_rec_int_ratio


data.total_rev_hi_lim=data.revol_bal/(data.revol_util/100)
data.total_rev_hi_lim.replace(np.inf,-1,inplace=1)


df_temp=data.groupby('batch_enrolled').agg({'loan_amnt':
                                            'mean'}).rename(columns={'loan_amnt':'batch_mean'}).reset_index()
data=pd.merge(data,df_temp,on='batch_enrolled',how='left')

df_temp=data.groupby('sub_grade').agg({'loan_amnt':
                                            'mean'}).rename(columns={'loan_amnt':'sg_mean'}).reset_index()
data=pd.merge(data,df_temp,on='sub_grade',how='left')

df_temp=pd.get_dummies(data['grade'],prefix='grade')
for col in df_temp.columns:
    data[col]=df_temp[col]
    
df_temp=pd.get_dummies(data['sub_grade'],prefix='sub_grade')
for col in df_temp.columns:
    data[col]=df_temp[col]
    
features=list(set(data.columns)-set(['emp_title','desc','title','verification_status_joint','loan_status',
                                     'member_id','total_loan','loan_left','loan_total'
                                     ]))
for col in features:
    lr=preprocessing.LabelEncoder()
    if data[col].dtype=='O':
        data[col].replace(np.nan,'AA',inplace=1)
        lr.fit(data[col])
        data[col]=lr.transform(data[col])
    data[col].replace(np.nan,-1,inplace=1)
    data[col].replace(np.inf,-1,inplace=1)
df_temp=data.groupby('batch_enrolled').agg({'last_week_pay':
                                             'value_counts'}).rename(columns={'last_week_pay':
                                                                              'week_counts'}).reset_index()

df_temp_1=df_temp.groupby('batch_enrolled').agg({'week_counts':'sum'}).rename(columns={'week_counts':
                                                                              'week_sum'}).reset_index()

df_temp=pd.merge(df_temp,df_temp_1,on='batch_enrolled',how='left')

df_temp.week_counts=df_temp.week_counts/df_temp.week_sum

data=pd.merge(data,df_temp[['batch_enrolled','last_week_pay','week_counts']],on=['batch_enrolled','last_week_pay'],
              how='left')
df_temp=data.last_week_pay.value_counts().reset_index().rename(columns={'last_week_pay':'week_ratio'})
df_temp.week_ratio=df_temp.week_ratio/df_temp.week_ratio.sum()

data['week_counts_remain']=1-data.week_counts      

train=data[~(data.loan_status.isnull())]
test=data[data.loan_status.isnull()]
test.drop('loan_status',axis=1,inplace=1)

df_temp=train.groupby('last_week_pay').agg({'loan_status':
                                            'mean'}).rename(columns={'loan_status':'fraction_default'}).reset_index()
train=pd.merge(train,df_temp,on='last_week_pay',how='left')
test=pd.merge(test,df_temp,on='last_week_pay',how='left')

features=list(set(train.columns)-set(['emp_title','desc','title','verification_status_joint','loan_status',

                                      'member_id','total_loan','batch_enrolled','loan_total','week_counts_remain',
                                      'tot_rec_int_ratio','interest_month','monthly_loan','inc_loan_ratio',
                                      'initial_list_status','loan_left','zip_code','pymnt_plan','tot_coll_amt','application_type'
                                      ,'funded_amnt_inv','purpose','delinq_2yrs','inq_last_6mths'
                                     ]))
                                     
total_index=train[train.loan_status==0].index

sampled_train1=train.iloc[np.random.choice(total_index,len(train[train.loan_status==1]),replace=False)]
sampled_train1=sampled_train1.append(train[train.loan_status==1])
sampled_train1=sampled_train1.iloc[np.random.permutation(len(sampled_train1))]

total_index=list(set(total_index)-set(list(sampled_train1.index)))                                    
