
# reference:
#https://www.kaggle.com/code/drisrarahmad/youtube-comment-dataset-nlp-ipynb
#https://medium.com/@mistrytejasm/text-preprocessing-removing-punctuation-and-special-characters-e3de4cece082
#
#

import pandas as pd
import re
from langdetect import detect
import nltk

#stopwords
from nltk.corpus import stopwords
nltk.download('stopwords')

#lemmatize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

#label encoder
from sklearn.preprocessing import LabelEncoder

# implement BoW count vectorizer
from sklearn.feature_extraction.text import CountVectorizer

# implement TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# test train split
from sklearn.model_selection import train_test_split 
  

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier 

# svm
from sklearn.svm import SVC

# naive bayes
from sklearn.naive_bayes import MultinomialNB

# accuracy
from sklearn.metrics import accuracy_score


def remove_punc(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

clean_stop = list(map (remove_punc,stopwords.words('english')))

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# thanks, https://www.kaggle.com/code/drisrarahmad/youtube-comment-dataset-nlp-ipynb
def remove_stopwords(text):
    # clean out punc from stopwords
    
    #print (clean_stop)
    
    new_text = []

    for word in text.split():
        if word in clean_stop:
            new_text.append('')
        else:
            new_text.append(word)
    x = new_text[:]
    new_text.clear()
    return " ".join(x)

def run_models():

    # test train split
    #x_bow_train, x_bow_test, y_train, y_test = train_test_split(x_bow,y, random_state=104,test_size=0.25,shuffle=True) 
    x_bow_train, x_bow_test, x_tf_train, x_tf_test, y_train, y_test = train_test_split(x_bow,x_tf,y,test_size=0.2,shuffle=True) 

    print ("shapes")
    print ("BoW training shape: ",x_bow_train.shape)
    print ("BoW test shape: ",x_bow_test.shape)
    print ("TF-IDF training shape: ",x_tf_train.shape)
    print ("TF-IDF test shape: ",x_tf_test.shape)
    print ("response training shape: ",y_train.shape)
    print ("response test shape: ",y_test.shape)

    # randomforest classifier BoW ---------------------------------------------------

    print ('.')
    print ('.')
    print ("running BoW/RF...")
    rf = RandomForestClassifier(n_estimators=10)
    rf.fit(x_bow_train,y_train)
 
    # prediction from the model
    y_pred = rf.predict(x_bow_test)
    # Score It

    print('Bag of Words/RandomForest Classifier:')
    # Accuracy

    bowrf_accuracy = round(accuracy_score(y_test, y_pred) * 100,2)
    print('Accuracy', bowrf_accuracy,'%')
    print('--'*30)
    #----------------------------------------------------------------------------

    # naive bayes BoW ---------------------------------------------------------------

    print ('.')
    print ('.')
    print ("running BoW/NB...")
    bowmnb = MultinomialNB()
    bowmnb.fit(x_bow_train,y_train)

    # prediction from the model
    y_pred = bowmnb.predict(x_bow_test)
    # Score It

    print('Bag of words/multinomial naive bayes Classifier: ')
    # Accuracy

    bowmnb_accuracy = round(accuracy_score(y_test, y_pred) * 100,2)
    print('naive bayes Accuracy', bowmnb_accuracy,'%')
    print('--'*30)
    # ---------------------------------------------------------------------------

    print ('.')
    print ('.')
    print ('.')
    print ('tf-idf')
    #randomforest classifier tf ---------------------------------------------------

    print ('.')
    print ('.')
    print ("running TF-IDF/RF...")
    rf2 = RandomForestClassifier(n_estimators=10)
    rf2.fit(x_tf_train,y_train)

    # prediction from the model
    y_pred = rf2.predict(x_tf_test)
    # Score It

    print('TF-IDF/RandomForest Classifier:')
    # Accuracy

    tfrf_accuracy = round(accuracy_score(y_test, y_pred) * 100,2)
    print('Accuracy', tfrf_accuracy,'%')
    print('--'*30)
    #----------------------------------------------------------------------------

    # naive bayes ---------------------------------------------------------------


    print ('.')
    print ('.')
    print ("running TF-IDF/RF...")
    tfmnb = MultinomialNB()
    tfmnb.fit(x_tf_train,y_train)
    
    # prediction from the model
    y_pred = tfmnb.predict(x_tf_test)
    # Score It

    print('TF-IDF/multinomial naive bayes Classifier:')
    # Accuracy

    tfmnb_accuracy = round(accuracy_score(y_test, y_pred) * 100,2)
    print('naive bayes Accuracy', tfmnb_accuracy,'%')
    print('--'*30)
    # ---------------------------------------------------------------------------

    return (bowrf_accuracy,bowmnb_accuracy,tfrf_accuracy,tfmnb_accuracy)



while True:
    print ("choose dataset: 1,2,3,4,5")
    C = input ("?")
    if int(C) in range (1,6):
           break
C=int(C)


# LOAD DATASET
#df=pd.read_csv("Y.csv")  # small ds
if C == 1:
    df=pd.read_csv("YoutubeCommentsDataSet.csv")  # big ds
    y_resp = 'Sentiment'
    x_corpus = 'Comment'
    print ("processing Youtube dataset")
#chatgpt
if C == 2:
    df=pd.read_csv("chatgpt.csv")
    df = df.sample(frac=0.1, random_state=42)
    y_resp = 'labels'
    x_corpus = 'tweets'

#financial
if C == 3:
    df=pd.read_csv("financial.csv")
    y_resp = 'Sentiment'
    x_corpus = 'Sentence'

#imdb
if C == 4:
    df=pd.read_csv("movie.csv")
    df = df.sample(frac=0.1, random_state=42)
    y_resp = 'label'
    x_corpus = 'text'

#twitter
if C == 5:
    df=pd.read_csv("twitter_training.csv")
    df = df.sample(frac=0.1, random_state=42)
    df.columns = ['A', 'B','label','text']
    y_resp = 'label'
    x_corpus = 'text'


print ("number of null entries:  ",df.isnull().sum())


#DROP DUPLICATES
print ("dropping duplicates...")
df.dropna(inplace=True)
df.drop_duplicates(subset=[x_corpus],inplace=True)
print ("number of null entries:  ",df.isnull().sum())


# CHECK DUPLICATED AND SEE HOW MANY SENTIMENT
print ("df y value counts: ")
print (df[y_resp].value_counts())
print ("number of duplicates: ",df[x_corpus].duplicated().sum())


print ("normalize, clean punctuation and stopwords...")
#lowercase normalize

df[x_corpus] = df[x_corpus].str.lower()



# CLEAN OUT PUNCTUATION
df[x_corpus]=df[x_corpus].apply(remove_punc)


# CLEAN OUT STOPWORDS
df[x_corpus] = df[x_corpus].apply(remove_stopwords)


#lang detect then add language column as 3rd column
print ("detect language...")
df["language"] = df[x_corpus].apply(detect_language)

print ("number of null: ",df.isnull().sum())
print ("lang value counts: ",df['language'].value_counts())


# subset only english language the drop language column (3r column) and save as df2
df = df[df['language'] == 'en']
df = df.drop('language', axis=1)
print ("dropped non-english entries...")
print ("df shape",df.shape)


# LEMMATIZATION
#https://www.geeksforgeeks.org/python-lemmatization-with-nltk/
print ("lemmatize...")
x = df[x_corpus]

count = 0
x_lem=[]
for i in x:
    count=count+1
    lem_list = []
    for j in i.split():
        lem_j=lemmatizer.lemmatize(j)
        lem_j2=lemmatizer.lemmatize(lem_j,pos="v")
        lem_j3=lemmatizer.lemmatize(lem_j2,pos="a")
        lem_j4=lemmatizer.lemmatize(lem_j3,pos="r")
        lem_list.append(lem_j4)
    lem_i=' '.join(lem_list)
    x_lem.append (lem_i)

print ("encoding response vector...")
#encode Sentiment label 'negative' as 0, neutral as '1', and positive as '2'
encoder = LabelEncoder()
df[y_resp] = encoder.fit_transform(df[y_resp])
y = df[y_resp]

print ("BoW Vectorization...")
#count vectorize
cv = CountVectorizer()
#x_bow = cv.fit_transform(x).toarray() # unlemmatized
x_bow = cv.fit_transform(x_lem).toarray()

print ("TF-IDF vectorization...")
tf = TfidfVectorizer()
#x_tf = tf.fit_transform(x).toarray() # unlemmatized
x_tf = tf.fit_transform(x_lem).toarray() # lemmatized

print ("XXXXXXXX")
print ("y shape: ",y.shape)
print ("x_bow shape: ",x_bow.shape)
print ("x_tf shape: ",x_tf.shape)


# https://www.geeksforgeeks.org/using-countvectorizer-to-extracting-features-from-text/
bowRF=[]
bowNB=[]
tfRF=[]
tfNB=[]
for i in range (100):
    bowRF_acc, bowNB_acc, tfRF_acc, tfNB_acc = run_models()
    print (bowRF_acc, bowNB_acc, tfRF_acc, tfNB_acc)
    bowRF.append(bowRF_acc)
    bowNB.append(bowNB_acc)
    tfRF.append(tfRF_acc)
    tfNB.append(tfNB_acc)

#print (bowRF, bowNB, tfRF, tfNB)
    
dfdict = {"bowRF": bowRF, "bowNB": bowNB, "tfRF": tfRF, "tfNB": tfNB}

result_df = pd.DataFrame(dfdict)
result_df.to_csv ('results.csv')




