import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Title
st.title("Restaurant Reviews Classification")

# Upload Dataset
uploaded_file = st.file_uploader("Upload a TSV file", type="tsv")
if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file, delimiter='\t', quoting=3)
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # Preprocessing
    st.write("### Preprocessing the Reviews")
    corpus = []
    ps = PorterStemmer()
    for i in range(0, len(df)):
        review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
    
    # Bag of Words
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(corpus).toarray()
    y = df.iloc[:, 1].values

    # Splitting dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Select Classifier
    st.sidebar.header("Model Selection")
    classifiers = {
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3, weights='distance'),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Bernoulli Naive Bayes": BernoulliNB(),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(),
        "Random Forest": RandomForestClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "LightGBM": LGBMClassifier(),
        "XGBoost": XGBClassifier(),
    }

    classifier_name = st.sidebar.selectbox("Choose Classifier", list(classifiers.keys()))
    classifier = classifiers[classifier_name]

    # Train the model
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    bias = classifier.score(X_train, y_train)
    variance = classifier.score(X_test, y_test)

    # Display Metrics
    st.write(f"### Results for {classifier_name}")
    st.write("Confusion Matrix:")
    st.write(cm)
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Bias (Training Score): {bias:.2f}")
    st.write(f"Variance (Test Score): {variance:.2f}")

    # Plot Confusion Matrix
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap='Blues')
    plt.title('Confusion Matrix', pad=20)
    plt.colorbar(cax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(fig)

else:
    st.write("Please upload a dataset to begin.")
