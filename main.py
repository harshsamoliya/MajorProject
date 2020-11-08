import pandas as pd
df = pd.read_csv('G:\python projects\FinallyMajorProject\Book1.csv')


# meathod 1 with simple analysis
import string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vs = SentimentIntensityAnalyzer()

# convering in lower case
df['review'] = df['review'].apply(lambda x : x.lower())

#  removing punctuation
def removerpunct(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

df['review'] = df['review'].apply(lambda  x : removerpunct(x))

# predicting values of vader

df['compound_pred']= df['review'].apply(lambda  x: vs.polarity_scores(x)['compound'])
#  seprating negative and postive value with variable pos and neg

df['pred']= df['compound_pred'].apply(lambda  x: 'positive' if x>0 else 'negative')
# print(df)

from sklearn.metrics import accuracy_score,confusion_matrix

# print(accuracy_score(df['label'],df['pred']))


x = df.iloc[:,0].values
y= df.iloc[:,1].values


# meathod 2 sklearn with support vector machine
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,random_state=0)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

final = Pipeline([('vect',TfidfVectorizer()),('model',SVC())])
final.fit(x_train,y_train)

y_pred = final.predict(x_test)
# print(accuracy_score(y_test,y_pred))

# deployment a model through stream-lit
import streamlit as st

st.title('Welcome to my project')
st.subheader('created by harsh samoliya')
message = st.text_area("Enter Text","Type Here...")
op = final.predict([message])
if st.button('predict'):
    st.title(op)
