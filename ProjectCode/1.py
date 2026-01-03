# %%
from joblib import dump
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
# %%
news_df = pd.read_csv('WELFake_Dataset.csv')
# %%
news_df = news_df[['text', 'label']]
# %%
news_df_cleaned = news_df.dropna()
# %%
y = news_df_cleaned['label']
X_train, X_test, y_train, y_test = train_test_split(
    news_df_cleaned['text'], y, test_size=0.20, random_state=19)
# %%
tfidf_vect = TfidfVectorizer(stop_words="english",max_df=0.7)

X_train_tfidf = tfidf_vect.fit_transform(X_train)
X_test_tfidf = tfidf_vect.transform(X_test)
# %%
# %%
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
news = ["The left believes these are all perfectly acceptable topics to discuss with our young children. Whatever you do, just don t mention God!A public hearing is taking place Wednesday morning in the Massachusetts State House to look into a controversial sex survey given to middle school and high school students.Developed by the Centers for Disease Control and called the  Youth Risk Behavior Survey,  the survey asks students as young as 12 a series of very personal and highly ideological questions.The survey asks students if they are homosexual and if they are transgender. It also asks if they have had oral or anal sex and if they have performed such acts with up to six people.Whether or not they have carried a gun, smoked cigarettes, consumed alcohol and how much also appear on the questionnaire, as well as whether they have taken drugs, such as OxyContin, Percocet, and Vicodin. It asks how often their guardian uses a seat belt, if the youngster has a sexually transmitted disease, and where they sleep.The group MassResistance says the survey is  psychologically distorting  and will lead the child to think he is  abnormal if he is not doing it all.  The group stated that  having children reveal personal issues about themselves and their family can have emotional consequences.  They also complain that  the survey results are used by radical groups from Planned Parenthood to LGBT groups to persuade politicians to give more taxpayer money [to] these groups. Though students fill out the survey anonymously, MassResistance warns that  they are administered by the teacher in the classroom and there is often pressure for all kids to participate. The test is given nationally and not without controversy. The Chicago Tribune reported two years ago that a Chicago teacher was reprimanded for telling students they had a  constitutional right  not to fill out the survey.Via: Breitbart News"]
#%%
news_tfidf = tfidf_vect.transform(news)
prediction = model.predict(news_tfidf)
print(prediction)
#%%
#%%
dump(model, 'model.pkl')
dump(tfidf_vect, 'vectorizer.pkl')
# %%
