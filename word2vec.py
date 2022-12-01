import pandas as pd
import gensim
import nltk
import numpy as np
import re
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import sent_tokenize
from gensim.utils import simple_preprocess

df = pd.read_csv('Cleaned_IMDB_Data.csv')
df = df[:1000]
X = df['review']
Y = df['Class']

story = []
for doc in df['review']:
    raw_sent = sent_tokenize(doc)
    for sent in raw_sent:
        story.append(simple_preprocess(sent))

model = gensim.models.Word2Vec(
    window=10,
    min_count=2
)
model.build_vocab(story)
model.train(story, total_examples=model.corpus_count, epochs=model.epochs)

def document_vector(doc):
    doc = [word for word in doc.split() if word in model.wv.index_to_key]
    return np.mean(model.wv[doc], axis=0)

X = []
for doc in tqdm(df['review'].values):
    X.append(document_vector(doc))
X = np.array(X)
Y = df['Class']      
X_Train,X_Test,Y_Train,Y_Test = train_test_split(X,Y,test_size=0.2,random_state=1)

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

Lg = LogisticRegression()
Lg.fit(X_Train,Y_Train)
pickle.dump(Lg, open('word2vec.pkl', 'wb'))
def predict(sen):
    sen = remove_tags(sen)
    sen = sen.lower()
    sen = omk(sen)
    sen = fun(sen)
    sen=document_vector(sen)
    # text = vect.transform([sen])
    pickled_model = pickle.load(open('word2vec.pkl', 'rb'))
    return pickled_model.predict([sen])
    # Lg = LogisticRegression()
    # Lg.fit(X_Train,Y_Train)
    # return Lg.predict([sen])[0]