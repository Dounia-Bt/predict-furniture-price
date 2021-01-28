# Importing the libraries
import pandas as pd
from sklearn import preprocessing
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

### Import data ###

df = pd.read_csv('furniture.csv', names=['item_id','name','category','old_price','sellable_online'
                            ,'link','other_colors','short_description','designer','depth'
                            ,'height','width','price'],skiprows=1, header=None)


# print('The shape of our data is:', df.shape)




### Preprocessing ###
categs = ['category','sellable_online','other_colors','depth','height','width','price']
furniture_data = df[categs]

# print('The shape of our new data is:', furniture_data.shape)

# Showing missing values
furniture_data.isnull().sum()

# Replacing the missing values with the mode value in the column
col_names = furniture_data.columns
for c in col_names:
    furniture_data = furniture_data.replace("?", np.NaN)
furniture_data = furniture_data.apply(lambda x:x.fillna(x.value_counts().index[0]))

# Verification of missing values being replaced
furniture_data.isnull().sum()

# Converting categorical values to numerical ones
cat_col=['category','sellable_online','other_colors']
labelEncoder = preprocessing.LabelEncoder()
mapping_dict ={}
for col in cat_col:
    furniture_data[col] = labelEncoder.fit_transform(furniture_data[col])
    le_name_mapping = dict(zip(labelEncoder.classes_,labelEncoder.transform(labelEncoder.classes_)))
    mapping_dict[col]= le_name_mapping
print(mapping_dict, "\n" )


### Extracting features

# Extracting data for fitting
X = furniture_data.drop('price', axis=1)  # features
y = furniture_data['price'] # labels 


### Training the model ###

# Spliting the data into training and test data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# print('X_train shape is', X_train.shape)
# print('X_test shape is', X_test.shape)
# print('y_train shape is', y_train.shape)
# print('y_test shape is', y_test.shape)


# #### Linear Regression

from sklearn.linear_model import LinearRegression

LRregressor = LinearRegression()
LRregressor.fit(X_train,y_train)

# Calculate the training and test score
print("LRregressor / Train score is :", LRregressor.score(X_train,y_train))
print("LRregressor / Test score is :", LRregressor.score(X_test,y_test))

# Calculate the prediction of the model
ypred_LR = LRregressor.predict(X_test)
print("Prediction of Linear Regression:", ypred_LR)

print('-------------------------------------------------------------------------')

# #### Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor

DTregressor = DecisionTreeRegressor(max_depth=10, random_state=1)
DTregressor.fit(X_train,y_train)

# Calculate the training and test score
print("Decision Tree Regressor / Train score is :", DTregressor.score(X_train,y_train))
print("Decision Tree Regressor / Test score is :", DTregressor.score(X_test,y_test))


# Calculate the prediction of the model
ypred_DTr = DTregressor.predict(X_test)
print("Prediction of Decision Tree:", ypred_DTr)

print('-------------------------------------------------------------------------')

# #### Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor

RFregressor = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=1)
RFregressor.fit(X_train,y_train)

# Calculate the training and test score
print("Random Forest Regressor / Train score is :", RFregressor.score(X_train,y_train))
print("Random Forest Regressor / Test score is :", RFregressor.score(X_test,y_test))
print("Random Forest Regressor / N° of features  are : ", RFregressor.n_features_)
print("Important features are : " , RFregressor.feature_importances_)


# Calculate the prediction of the model
ypred_RFr = RFregressor.predict(X_test)
print("Prediction of Random Forest :", ypred_RFr)

print('-------------------------------------------------------------------------')

# #### SVM Regressor

from sklearn.svm import SVR

SVMregressor = SVR(kernel='linear')
SVMregressor.fit(X_train, y_train)

# Calculate the training and test score
print("SVM regressor / Train score is :", SVMregressor.score(X_train,y_train))
print("SVM regressor / Test score is :", SVMregressor.score(X_test,y_test))


# Calculate the prediction of the model
ypred_SVMr = SVMregressor.predict(X_test)
print("Prediction of SVM :", ypred_SVMr)


### Saving the best model ###

# Saving model to disk
pickle.dump(DTregressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
# print("Result of prediction:",model.predict(X_test))

