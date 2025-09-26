import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')
stop_words = stopwords.words('spanish')

dataframe = pd.read_csv("./datasets/ds_llm.csv") 

print(dataframe.head())

dataframe = dataframe.dropna(subset=['text'])

vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(dataframe['text'])
y = dataframe['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = MultinomialNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

feature_names = vectorizer.get_feature_names_out()
coefs = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
top_spam = np.argsort(coefs)[-10:]
top_ham = np.argsort(coefs)[:10]
print("Palabras mas asociadas a spam:", feature_names[top_spam])
print("Palabras mas asociadas a ham:", feature_names[top_ham])



print("\nEscribe un mensaje y el programa te dirá si es spam o ham.")
print("Escribe 'salir' para terminar.\n")

while True:
    mensaje = input("Mensaje: ")
    if mensaje.lower() == 'salir':
        print("Saliendo...")
        break

    mensaje_vec = vectorizer.transform([mensaje])
    prediccion = clf.predict(mensaje_vec)[0]
    print("Clasificación:", "spam" if prediccion == 1 else "ham")
    print("---")
