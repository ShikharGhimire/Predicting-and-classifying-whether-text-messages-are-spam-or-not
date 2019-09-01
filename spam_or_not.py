#Natural language processing 

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Looking at the dataset the first coloumn is the dependent variable explaining whether the text message is ham or spam and the second coloumn is the text itself
#First  converting csv file to tsv file
#Use this link :https://onlinetsvtools.com/convert-csv-to-tsv 
#Importing the dataset
dataset = pd.read_csv('spam.tsv', delimiter = '\t',quoting = 3) #Note that csv file was converted to tsv because in csv
#data are seperated by commas and since reviews itself will have commas
# therefore the algorithm will get confused and algorithms doesn't work 
#Quoting = 3 ignores double quotes in reviews as it is not necessary  or helpful to distinguish
#Cleaning the text in the text message
#Cleaning the text is important as some words won't be very helpful in determining and classifying between ham and spam
#message 

#Assigning the dependent variable
y = dataset.iloc[:,0]
#Label encode the dependent variable
from sklearn.preprocessing import LabelEncoder
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)
#1 indicates the text message is spam and 0 for the ham messages

import re #cleaning the text
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords #Implementing the stopwords
from nltk.stem.porter import PorterStemmer #creating a root word meaning changing past and future tense to present tense. Example "loved" will be "love"

corpus=[] #corpus to store the words
#Cleaning and preparing the data 
for i in range(0,5559):
    text_message = re.sub('[^a-zA-Z]',' ' ,dataset['text'][i])# '^' represents any letter that we don't want to remove while taking  care of the spaces in between
    text_message = text_message.lower() #Setting everything in lower cases
    text_message = text_message.split() #Splitting the whole sentence into words
    ps = PorterStemmer()
    text_message = [ps.stem(word) for word in text_message if not word in set (stopwords.words('english'))]
    #In above it first performs 'stopwords' and then performs stemming on the output of the loop 'word'
    text_message = ' '.join(text_message) #Reverting it back to non list strings 
    corpus.append(text_message)

#Each row in the matrix represents a review
#Each coloumn in the matrix represents a single word from the reviews
#Creating a bag of models
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 6000) #Keeping 6000 must used words
X = cv.fit_transform(corpus).toarray()#Independent variable

#Splitting the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Using naive bayes to train in the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

#Predicting the test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred) #962 correct predicition #150 incorrect prediciton 

