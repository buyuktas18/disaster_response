<<<<<<< HEAD
# disaster_response
An ETL and NLP pipeline which extracts meaning from disaster texts
=======
# Disaster Response Pipeline Project

## Purpose
In this project, the aim is to help people and charities in case of emergency. How it will do it? It will receive a text which is written during or just after the disaster and try to match the topics belonging to that text. By using this application, foundations or charities could extract the amount of need and the situation of the disaster. Since these parameters would be more visible, people who need help could get the response faster.

The purpose of this project is to classify the main needs according to disaster tweets. This will be a multiclass classification since a tweet could be belong to several topics. Basic NLP pipeline has been constructed to solve this problem. The main points in this pipeline can be briefly listed as:
- ETL pipeline
	- Data extracting
    - Data Cleaning
    - Data Storage in SQL database
- NLP pipeline
	- Tokenize
    - Lemmatize
    - Count Vectorizer
    - TF-IDF
- ML pipeline
	- Random Forest Classifier
    - Evaluation Metrics
    
### Files 

- **data/process.py:** This file for processing data. It reads messages and categories from seperate csv files, combines them by using some transformations and keep the organised result in SQL database. 

- **model/classifier.py:** This file is using for creating the machine learning model and also apply some NLP functions such as tokenizer and tf-idf. A pipeline is defined in this file to work with model easily. Then the trained model is saved as .pkl file.

- **data/messages.csv:** This file keeps the records of messages
- **data/categories.csv:** This file keeps the records of categories

- **app/run.py:** This is the web application created by Flask

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
>>>>>>> master
