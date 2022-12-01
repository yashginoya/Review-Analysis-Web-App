import pandas as pd
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
# from sklearn.linear_model import LogisticRegression

df = pd.read_csv('Cleaned_IMDB_Data.csv')
X = df['review']
Y = df['Class']
X_Train,X_Test,Y_Train,Y_Test = train_test_split(X,Y,test_size=0.2,random_state=40,stratify=Y)
X_Train = X_Train.reset_index()['review']
X_Test = X_Test.reset_index()['review']
Y_Train = Y_Train.reset_index()['Class']
Y_Test = Y_Test.reset_index()['Class']

vect = CountVectorizer()
X_Train = vect.fit_transform(X_Train)
X_Test = vect.transform(X_Test)

ps=PorterStemmer()
def fun(text):
    y=[]
    for i in text.split():
      y.append(ps.stem(i))
    return " ".join(y)

def remove_tags(raw_text):
      cleaned_text = re.sub(re.compile('<.*?>'), '', raw_text)
      return cleaned_text

sw_list = stopwords.words('english')
def omk(sentence):
      opp=[]
      for item in sentence.split():
            if item not in sw_list:
               opp.append(item)

      return " ".join(opp)

Lg = MultinomialNB()
Lg.fit(X_Train,Y_Train)
pickle.dump(Lg, open('bagOfWords.pkl', 'wb'))
def predictText(sen):
      sen = remove_tags(sen)
      sen = sen.lower()
      sen = omk(sen)
      sen = fun(sen)
      text = vect.transform([sen])
      pickled_model = pickle.load(open('bagOfWords.pkl', 'rb'))
      return pickled_model.predict(text)
      # print(Lg.predict(vect.transform([sen]))[0])


# text = input("Enter the text: ")
# print(predict(text))


# text = vect.transform([text])
#     pickled_model = pickle.load(open('model.pkl', 'rb'))
#     return pickled_model.predict(text)