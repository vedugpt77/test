import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import warnings
warnings.filterwarnings('ignore')
####################################################################
f = open('DP.txt','r')
raw = f.read()
raw = raw.lower()
####################################################################
sent_tokenize = nltk.sent_tokenize(raw)
joblib.dump(sent_tokenize, 'st.pkl')
word_tokenize = nltk.word_tokenize(raw)
####################################################################
lem = nltk.stem.WordNetLemmatizer()
def lemTokens(tokens):
    return [lem.lemmatize(token) for token in tokens]

PunctDict = dict((ord(punct),None) for punct in string.punctuation)

def toknzr(text):
    return lemTokens(nltk.word_tokenize(text.lower().translate(PunctDict)))
####################################################################
def response(user_response):
    resp = ''
    st = joblib.load('st.pkl')
    st.append(user_response)
    tfidfvec = TfidfVectorizer()
    tfidf = tfidfvec.fit_transform(st)
    sim_mat = cosine_similarity(tfidf[-1],tfidf)
    id = sim_mat.argsort()[0][-2]
    f = sim_mat.flatten()
    f.sort()
    req_id = f[-2]
    if req_id==0:
        resp = 'Sorry I didn\'t understand you.'
    else:
        resp =  st[id]
    return resp
