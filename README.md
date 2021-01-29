# predict-furniture price
This is a short tutorial to deploy a machine learning model on heroku with flask and use Apache Airflow to author, schedule and monitor workflows.
This project is created to predict the price of the furniture based on a machine learning model.

## Project Structure
The project contains 3 main files:
1. furniturePred.py : contains the code of our Machine Learning model to predict furniture price.
2. app.py : runs the Flask application.
3. pipeline_dag.py : contains the dags that will be executed by Airflow.

## Step by step
### Requirements
To run this project locally follow the steps below :

* Clone the repository and open the project in PyCharm.

* Install the requirements on *requirements.txt* by entering this command on the terminal :
```
$ pip install -r requirements.txt
```
### Flask app
* Run furniturePred.py
```
$ python furniturePred.py
```
It creates the model.pkl file.

* Run app.py using below command to start Flask API
```
$ python app.py
```
By default, flask will run on port 5000.

### Deployment on Heroku

Finally to deploy the project on Heroku follow these commands :

* Create a Heroku account if you don't have one.

* Install Heroku CLI.

* Inside the PyCharm terminal enter the following commands :
```
$ heroku login 
$ heroku create AppName
$ git init 
$ git add .
$ git commit -am "commit message"
$ git push heroku master
```
At the end you will have the heroku link : https://dounia-predict-furniture-price.herokuapp.com/

![](https://github.com/DouniaBtb/predict-furniture-price/blob/main/heroku-app-prediction.PNG)

### Apache Airflow

Install Airflow into Windows 10 WSL with Ubuntu or install Airflow directly on Ubuntu or in Pycharm with the following command :
```
$ pip intsall apache-airflow (or pip3 for Python 3)
```

Then initialize the SQLite database with :
```
$ airflow db init
```

Once you are done installing, edit airflow.cfg by making the following changes :

* dags_folder = path of the dags folder (use **/mnt/path_of_the_dags_folder** if you're working with the Windows 10 WSL)
* load_examples = False
* enable_xcom_pickling = True

At each change, the database must be reinitialized.


For more details, click here : [Quick Start Apache Airflow](https://airflow.apache.org/docs/apache-airflow/1.10.11/start.html)


Then open two separate terminals and run on each terminal the following commands :
```
$ airflow webserver 
```
```
$ airflow scheduler
```

Navigate to the default URL http://0.0.0.0:8080/ , switch on the DAG and go to Graph View.


![](https://github.com/DouniaBtb/predict-furniture-price/blob/main/airflow-dag.png)




