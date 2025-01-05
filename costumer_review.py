import pandas as pd
import numpy as np 
# loading dataset
df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\NIT All-Projects\Costumer Review\Restaurant_Reviews.tsv",delimiter='\t', quoting = 3)
# doubling the dataset to increase the accuracy of the model
doubled_df = pd.concat([df, df], ignore_index=True)

# Save the doubled dataset if needed
doubled_df.to_csv("doubled_dataset.tsv", sep='\t', index=False)

# Print the result to verify
print("Original dataset size:", df.shape)
print("Doubled dataset size:", doubled_df.shape)
import re 
import nltk
#ntlk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range (0,1000):
    review = re.sub('[^a-zA-Z]',' ',df['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word)for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
#creating the bag of words model 
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:,1].values

# Fill NaN values with the median stratgy
#df[:,1].fillna(df[:,1].median(), inplace=True)
#df.fillna(df.mean(),inplace=True)
#df['Liked'].fillna(df['Liked'].median(), inplace=True)

#splitting the dataset into the training set and set test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02,random_state=0)

#decision tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)

#KNN
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=3,weights='distance')
classifier.fit(X_train,y_train)

#Multinomial
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
classifier.fit(X_train,y_train)

#Bernouli
from sklearn.naive_bayes import BernoulliNB
classifier=BernoulliNB()
classifier.fit(X_train,y_train)

#GaussianNB
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)

#SVM
from sklearn.svm import SVC
classifier=SVC()
classifier.fit(X_train,y_train)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)

#Adaboost
from sklearn.ensemble import AdaBoostClassifier
classifier=AdaBoostClassifier()
classifier.fit(X_train,y_train)

#lgbm
from lightgbm import LGBMClassifier
classifier=LGBMClassifier()
classifier.fit(X_train,y_train)

#xgboost
from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(X_train,y_train)

#predicting the Test set results
y_pred = classifier.predict(X_test)

#making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#accuracy of the model
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

#bias score
bias = classifier.score(X_train, y_train)
print(bias)

#variance score
variance = classifier.score(X_test, y_test)
print(variance)
