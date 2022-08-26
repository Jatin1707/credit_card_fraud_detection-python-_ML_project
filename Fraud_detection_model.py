import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading the dataset toaPandas DataFrame
cred_card_data = pd.read_csv("creditcard.csv")
cred_card_data.head()
cred_card_data.tail()
 
# credit card info

cred_card_data.info()

# checking for missing values in a dataset

cred_card_data.isnull().sum()

# distribution of lagit and fraud transactions 

print(cred_card_data['Class'].value_counts())

''' this dataset is highly unbalanced so we need to fix it here --
    0 --> normal transactions
    1 --> fraudulent transactions '''
    
# separating the data for analysis

legit = cred_card_data[cred_card_data.Class == 0]    
fraud = cred_card_data[cred_card_data.Class == 1]   

#print(legit.shape)
#print(fraud.shape)

print(legit.Amount.describe())
print(fraud.Amount.describe())

#compare the values for both transactions

cred_card_data.groupby('Class').mean()

''' under sampling the dataset
    build a sample dataset contiaining 
    similar distribution of normal transaction and fraudulant transactons'''
    
# now we are taking a sample of legit dataset of 492 to match with fraud transactions 

legit_sample = legit.sample(n=492)   

# CONCATINATING TWO DATASET i.e., legit_sample and fraud dataset

new_dataframe = pd.concat([legit_sample,fraud],axis =0)     # axis=0 (add values row wise)
     
                                                            # axis =1 (add values col. wise)
       
new_dataframe.head()
                                                            
new_dataframe.groupby('Class').mean()

# Splitting the data into features and Targets

X = new_dataframe.drop(columns = 'Class',axis = 1)
Y = new_dataframe["Class"]

''' now spliting the data into Training data and Testing data'''

X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)

                # MODEL TRAINING

''' LOGISTIC REGRESSION '''

model = LogisticRegression()

#  training the logistic regression model with taining data

model.fit(X_train,Y_train)

                # EVALUATION MODEL
                # ACCURACY SCORE

''' ACCURACY ON TRAINING DATA '''

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)

print('Accuracy on Training data : ',training_data_accuracy)

''' ACCURACY ON TEST DATA '''

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)

print('Accuracy on Testing data : ',test_data_accuracy)





















 