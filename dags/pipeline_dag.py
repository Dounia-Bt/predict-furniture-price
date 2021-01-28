import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import numpy as np
import pandas as pd
import pandas as pd
from sklearn import preprocessing
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR



########## CHARGEMENT DES DONNEES ####################

def get_data(**context):

    data = pd.read_csv('/mnt/c/Users/hp/PycharmProjects/furniture_pred/dags/furniture.csv',encoding="ISO-8859-1")
    context['task_instance'].xcom_push(key='dataframefurniture',value=data)
    data.head()
    data.info()
    return data

########## PREPROCESSING ############

def data_preprocessing(**context):
    df= context['task_instance'].xcom_pull(task_ids='get_data',key='dataframefurniture')
    categs = ['category', 'sellable_online', 'other_colors', 'depth', 'height', 'width', 'price']
    furniture_data = df[categs]

    print('The shape of our new data is:', furniture_data.shape)

    # Showing missing values
    print("Missing Values " ,furniture_data.isnull().sum())

    # Replacing the missing values with the mode value in the column
    col_names = furniture_data.columns
    for c in col_names:
        furniture_data = furniture_data.replace("?", np.NaN)
    furniture_data = furniture_data.apply(lambda x: x.fillna(x.value_counts().index[0]))

    # Verification of missing values being replaced
    print("Missing Values after preprocessing",furniture_data.isnull().sum())

    # Converting categorical values to numerical ones
    cat_col = ['category', 'sellable_online', 'other_colors']
    labelEncoder = preprocessing.LabelEncoder()
    mapping_dict = {}
    for col in cat_col:
        furniture_data[col] = labelEncoder.fit_transform(furniture_data[col])
        le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
        mapping_dict[col] = le_name_mapping
    #print(mapping_dict, "\n")
    context['task_instance'].xcom_push(key='dataframefurnitureClean', value=furniture_data)
    return furniture_data

########### Features Selection #################

def features_selection(**context):
    furniture_data = context['task_instance'].xcom_pull(task_ids='data_preprocessing', key='dataframefurnitureClean')
    X = furniture_data.drop('price', axis=1)  # features
    y = furniture_data['price']  # labels
    # Spliting the data into training and test data sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    context['task_instance'].xcom_push(key='xtrain', value=X_train)
    context['task_instance'].xcom_push(key='ytrain', value=y_train)
    context['task_instance'].xcom_push(key='xtest', value=X_test)
    context['task_instance'].xcom_push(key='ytest', value=y_test)

########### REGRESSION LINEAIRE  #########
def linear_regression(**context):
    X_train=context['task_instance'].xcom_pull(task_ids='features_selection', key='xtrain')
    y_train=context['task_instance'].xcom_pull(task_ids='features_selection', key='ytrain')
    X_test=context['task_instance'].xcom_pull(task_ids='features_selection', key='xtest')
    y_test=context['task_instance'].xcom_pull(task_ids='features_selection', key='ytest')
    LRregressor = LinearRegression()
    LRregressor.fit(X_train, y_train)

    # Calculate the training and test score
    print("LRregressor / Train score is :", LRregressor.score(X_train, y_train))
    print("LRregressor / Test score is :", LRregressor.score(X_test, y_test))

    # Calculate the prediction of the model
    ypred_LR = LRregressor.predict(X_test)
    print("Prediction :", ypred_LR)

########## DECISION TREE REGRESSOR ##########

def decision_tree_regressor(**context):
    X_train = context['task_instance'].xcom_pull(task_ids='features_selection', key='xtrain')
    y_train = context['task_instance'].xcom_pull(task_ids='features_selection', key='ytrain')
    X_test = context['task_instance'].xcom_pull(task_ids='features_selection', key='xtest')
    y_test = context['task_instance'].xcom_pull(task_ids='features_selection', key='ytest')
    DTregressor = DecisionTreeRegressor(max_depth=10, random_state=1)
    context['task_instance'].xcom_push(key='bestmodel', value=DTregressor)
    DTregressor.fit(X_train, y_train)

    # Calculate the training and test score
    print("Decision Tree Regressor / Train score is :", DTregressor.score(X_train, y_train))
    print("Decision Tree Regressor / Test score is :", DTregressor.score(X_test, y_test))

    # Calculate the prediction of the model
    ypred_DTr = DTregressor.predict(X_test)
    print("Prediction :", ypred_DTr)


################## RANDOM FOREST REGRESSOR ##############

def random_forest_regressor(**context):
    X_train = context['task_instance'].xcom_pull(task_ids='features_selection', key='xtrain')
    y_train = context['task_instance'].xcom_pull(task_ids='features_selection', key='ytrain')
    X_test = context['task_instance'].xcom_pull(task_ids='features_selection', key='xtest')
    y_test = context['task_instance'].xcom_pull(task_ids='features_selection', key='ytest')
    RFregressor = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=1)
    RFregressor.fit(X_train, y_train)

    # Calculate the training and test score
    print("Random Forest Regressor / Train score is :", RFregressor.score(X_train, y_train))
    print("Random Forest Regressor / Test score is :", RFregressor.score(X_test, y_test))
    print("Random Forest Regressor / N° of features  are : ", RFregressor.n_features_)
    print("Important features are : ", RFregressor.feature_importances_)

    # Calculate the prediction of the model
    ypred_RFr = RFregressor.predict(X_test)
    print("Prediction :", ypred_RFr)

########### SVM #############

def SVM(**context):
    X_train = context['task_instance'].xcom_pull(task_ids='features_selection', key='xtrain')
    y_train = context['task_instance'].xcom_pull(task_ids='features_selection', key='ytrain')
    X_test = context['task_instance'].xcom_pull(task_ids='features_selection', key='xtest')
    y_test = context['task_instance'].xcom_pull(task_ids='features_selection', key='ytest')
    SVMregressor = SVR(kernel='linear')
    SVMregressor.fit(X_train, y_train)

    # Calculate the training and test score
    print("SVM regressor / Train score is :", SVMregressor.score(X_train, y_train))
    print("SVM regressor / Test score is :", SVMregressor.score(X_test, y_test))

    # Calculate the prediction of the model
    ypred_SVMr = SVMregressor.predict(X_test)
    print("Prediction :", ypred_SVMr)

def save_best_model(**context):
    DTregressor= context['task_instance'].xcom_pull(task_ids='decision_tree_regressor', key='bestmodel')
    pickle.dump(DTregressor, open('model.pkl', 'wb'))


############# DAG ###################

default_args = {
    'owner': 'airflow',
    'start_date': airflow.utils.dates.days_ago(2),
    'depends_on_past': False,
    'email': ['dounia.boutayeb97@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    # 'end_date': datetime(2018, 12, 30),
    #'retries': 1,
    # If a task fails, retry it once after waiting
    # at least 5 minutes
    #'retry_delay': timedelta(minutes=5),
    }


dag = DAG(
    dag_id= 'ML_pipeline',
    start_date= airflow.utils.dates.days_ago(2),
    default_args=default_args,
    description='Automatisation des modeles de ML',
    #schedule_interval=timedelta(days=1),
)


get_data = PythonOperator(
    task_id='get_data',
    python_callable = get_data,
    #xcom_push=True,
    #provide_context=True,
    dag=dag,
)

data_preprocessing = PythonOperator(
    task_id='data_preprocessing',
    python_callable =data_preprocessing ,
    #xcom_push=True,
    #provide_context=True,
    dag=dag,
)

features_selection = PythonOperator(
    task_id='features_selection',
    python_callable =features_selection ,
    #xcom_push=True,
    #provide_context=True,
    dag=dag,
)

linear_regression = PythonOperator(
    task_id='linear_regression',
    python_callable =linear_regression ,
    #xcom_push=True,
    #provide_context=True,
    dag=dag,
)

decision_tree_regressor = PythonOperator(
    task_id='decision_tree_regressor',
    python_callable =decision_tree_regressor ,
    #xcom_push=True,
    #provide_context=True,
    dag=dag,
)

random_forest_regressor = PythonOperator(
    task_id='random_forest_regressor',
    python_callable =random_forest_regressor,
    #xcom_push=True,
    #provide_context=True,
    dag=dag,
)

SVM = PythonOperator(
    task_id='SVM',
    python_callable =SVM,
    #xcom_push=True,
    #provide_context=True,
    dag=dag,
)
save_model = PythonOperator(
    task_id='savemodel',
    python_callable =save_best_model,
    #xcom_push=True,
    #provide_context=True,
    dag=dag,

)

### SCHEDULE ###########

get_data >> data_preprocessing
data_preprocessing >> features_selection
features_selection >> linear_regression
features_selection >> decision_tree_regressor
features_selection >> random_forest_regressor
features_selection >> SVM
decision_tree_regressor >> save_model