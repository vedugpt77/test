from flask import Flask, render_template, request, jsonify
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import warnings
warnings.filterwarnings('ignore')
########################################################
app = Flask(__name__)
########################################################
@app.route('/')
def home():
    return render_template('index.html')
########################################################
lem = nltk.stem.WordNetLemmatizer()
def lemTokens(tokens):
    return [lem.lemmatize(token) for token in tokens]

PunctDict = dict((ord(punct),None) for punct in string.punctuation)

def tk(text):
    return lemTokens(nltk.word_tokenize(text.lower().translate(PunctDict)))
########################################################
@app.route('/respond', methods=['POST'])
def respond():
    ans=''
    s_t = joblib.load('st.pkl')
    user_query = request.form.to_dict()
    s_t.append(user_query['question_text'])
    tfidf = TfidfVectorizer(tokenizer=tk, stop_words='english')
    tfidfvec = tfidf.fit_transform(s_t)
    sim_mat = cosine_similarity(tfidfvec[-1],tfidfvec)
    id = sim_mat.argsort()[0][-2]
    f = sim_mat.flatten()
    f.sort()
    req_id = f[-2]
    if req_id==0:
        ans = 'Sorry I didn\'t understand you.'
    else:
        ans =  s_t[id]

    return jsonify({'Answer' : ans})
########################################################
if __name__ == '__main__':
    app.run('0.0.0.0')
