from flask import Flask, request, Response, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Load raw data file
av_2019Survey = pd.read_csv("./Data/av_2019.csv", header=0)

#Handle missing values
categorical_columns = ["FamiliarityNews", "FamiliarityTech", "SharedCyclist", "SharedPedestrian", "AvImpact", "ProvingGround", "Speed25Mph", "TwoEmployeesAv", "ZipCode",
                       "SchoolZoneManual","ShareTripData","SharePerformanceData","Age","ReportSafetyIncident", "ArizonaCrash", "BikePghMember", "AutoOwner", "SmartphoneOwner"]
av_2019Survey[categorical_columns] = av_2019Survey[categorical_columns].fillna("Missing")

numeric_columns = ['SafeAv', 'SafeHuman']
for column in numeric_columns:
    mean_value = av_2019Survey[column].mean()
    av_2019Survey[column].fillna(mean_value, inplace=True)

# Top 8 variables are selected based on the value from correlation analysis
X_categoric_top8 = av_2019Survey.loc[:, ['FamiliarityTech','SharePerformanceData', 'ReportSafetyIncident',
                                    'ArizonaCrash','Speed25Mph','ProvingGround','AvImpact','SchoolZoneManual']].values

#Onehotencoding
ohe = OneHotEncoder()
categoric_data_top8 = ohe.fit_transform(X_categoric_top8).toarray()
categoric_df_top8 = pd.DataFrame(categoric_data_top8)
categoric_df_top8.columns = ohe.get_feature_names_out()
categoric_df_top8.head()

#Target variable
safe_av_df =av_2019Survey [["SafeAv"]]

#Combine target and independent variables 
df_2019_SafeAv = pd.concat([categoric_df_top8,safe_av_df], axis = 1)

#Logistic Regression
df = df_2019_SafeAv
df['SafeAv'] = df['SafeAv'].astype(float)

# Separate the features (X) and the target variable (y)
X = df.drop('SafeAv', axis=1)
y = df['SafeAv']

# Transform the target variable
y = y.apply(lambda x: 'Yes' if x >= 3 else 'No')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)

#create flask instance
app = Flask(__name__)

#create api which be hosted on the server
@app.route('/api', methods=['GET', 'POST'])  
def predict():
    data = request.get_json(force=True)        #get data from request
    data_categoric = np.array([data["FamiliarityTech"], data["SharePerformanceData"], data["ReportSafetyIncident"], data["ArizonaCrash"], data["Speed25Mph"], data["ProvingGround"],
                               data["AvImpact"],data["SchoolZoneManual"] ])
    data_categoric = np.reshape(data_categoric, (1, -1))   #reshape the array to a column

    data_categoric = ohe.transform(data_categoric).toarray()   #convert text to numeric data by using one hot encoder 

    data_final = pd.DataFrame(data_categoric, dtype=object)

    #make predicon using model
    prediction = model.predict(data_final)
    return Response(json.dumps(prediction[0])) # Response will be either Yes or No and sent back to requester 


