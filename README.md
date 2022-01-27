# Disaster Response Pipeline Project

## Project Overview:
The goal of this project is to build a machine learning model to identify if these messages are related to disaster or not, and further label the nature of these messages. This would be of great help for some disaster relief agencies. We have 36 labels for these messages in total. Note, however, these labels are not mutually exclusive. Hence it is a multi-label classification problem.

The most obvious feature of those data messages is they are highly imbalanced. Several categories getting very few labels.

After building and training such a model, we can next launch a web service which can label new messages from users' input.

## File Description:
- process_data.py: This python excutuble code takes as its input csv files containing message data and message categories (labels), and then creates a SQL database
- train_classifier.py: This code trains the ML model with the SQL data base
- data: This folder contains sample messages and categories datasets in csv format and Database for the cleaned data.
- app: This folder cointains the run.py to iniate the web app.


- app
├── template
├── master.html  # main page of web app
├── go.html  # classification result page of web app
└── run.py  # Flask file that runs app

- data
├── disaster_categories.csv  # data to process 
├── disaster_messages.csv  # data to process
├── process_data.py
└── InsertDatabaseName.db   # database to save clean data to

- models
├── train_classifier.py
└── classifier.pkl  # saved model 

- README.md

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
