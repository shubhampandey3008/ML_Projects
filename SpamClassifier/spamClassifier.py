# Imports
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

nltk.download('stopwords')  
# Reading csv and creating dataframe out of it using \t separator
messages = pd.read_csv("SMSSpamCollection.txt" , sep = '\t' , names = ["label" , "message"])



# DATA PREPROCESSING
ps = PorterStemmer()
corpus = []
# Data Preprocessing : Removing stopwords , non alphabetical characters
for i in range(0 , len(messages)):
    review = re.sub('[^a-zA-Z]' , ' ' , messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# Creating Bag of Words
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()
# Y = cv.fit_transform(corpus)

# Array of dependent variable , Creating labels for the model
y = pd.get_dummies(messages["label"])
y = y.iloc[:,1].values

X_train , X_test , Y_train , Y_test = train_test_split(X , y , train_size=0.2 , random_state=0)

# Using Naive Bayes algo to train the model
spam_detect_model = MultinomialNB().fit(X_train , Y_train)

# Testing the model using the testing data
y_pred = spam_detect_model.predict(X_test)
print(Y_test)


# Using confusion matrix to test our model
confusion_m = confusion_matrix(Y_test , y_pred)
print(confusion_m)


# Checking the accuracy score
accuracy = accuracy_score(Y_test , y_pred)
print(accuracy)