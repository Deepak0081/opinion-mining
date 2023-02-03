import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import gensim
from gensim.models.doc2vec import TaggedDocument
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import joblib
from tqdm import tqdm

# Data preprocessing
train = pd.read_csv('train_tweet.csv')
test = pd.read_csv('test_tweets.csv')

train['len'] = train['tweet'].str.len()
test['len'] = test['tweet'].str.len()

cv = CountVectorizer(stop_words = 'english')
words = cv.fit_transform(train.tweet)

#nltk.download('stopwords') #If this dosen't work manually copy the 'nltk_data' folder into C:\\ or /usr 

def hashtag_extract(x):
    hashtags = []
    
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags

HT_regular = hashtag_extract(train['tweet'][train['label'] == 0])
HT_negative = hashtag_extract(train['tweet'][train['label'] == 1])

HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])

tokenized_tweet = train['tweet'].apply(lambda x: x.split()) 

model_w2v = gensim.models.Word2Vec(
            tokenized_tweet,
            vector_size=200, 
            window=5, 
            min_count=2,
            sg = 1, 
            hs = 0,
            negative = 10, 
            workers= 2, 
            seed = 34)

model_w2v.train(tokenized_tweet, total_examples= len(train['tweet']), epochs=20)

tqdm.pandas(desc="progress-bar")

def add_label(twt):
    output = []
    for i, s in zip(twt.index, twt):
        output.append(TaggedDocument(s, ["tweet_" + str(i)]))
    return output


labeled_tweets = add_label(tokenized_tweet)
labeled_tweets[:6]

train_corpus = []
for i in range(0, 31962):
    review = re.sub('[^a-zA-Z]', ' ', train['tweet'][i])
    review = review.lower()
    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    train_corpus.append(review)


test_corpus = []
for i in range(0, 17197):
    review = re.sub('[^a-zA-Z]', ' ', test['tweet'][i])
    review = review.lower()
    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    test_corpus.append(review)


# Creating a bag of words for train
cv = CountVectorizer(max_features = 2500)
x = cv.fit_transform(train_corpus).toarray()
y = train.iloc[:, 1]

# Splitting data
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.25, random_state = 42)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_valid = sc.transform(x_valid)

# Training the model
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_valid)

filename = "Completed_model.joblib"
joblib.dump(model, filename)

# Model Score
print("Training Accuracy   :", model.score(x_train, y_train))
print("Validation Accuracy :", model.score(x_valid, y_valid))
print("F1 score            :", f1_score(y_valid, y_pred))


# Confusion matrix
cm = pd.DataFrame(confusion_matrix(y_valid, y_pred), columns= ['Positive', 'Negative'], index= [['Positive', 'Negative']])
ax = plt.axes()
sns.heatmap(cm, annot= True, fmt='g', cmap= 'mako', ax= ax)
ax.set_title("Confusion Matrix for Validation results")

# Pie chart
labels = ['Positive', 'Negative']
data = [len(y_pred) - np.sum(y_pred), np.sum(y_pred)]
colors = ['#00d904', '#fc0303',]
fig = plt.figure(figsize =(10, 7))
plt.pie(data, labels = labels, colors= colors)
plt.title("Distribution of positivie and negative comments in test dataset")
plt.show()

# Output data
for i in range(len(y_pred)):
    twt = train.loc[x_train.shape[0] - 1 + i]['tweet']
    print(f'{twt:<150}', end = "")
    if y_pred[i] == 0:
        print("POSITIVE")
    else:
        print("NEGATIVE")


