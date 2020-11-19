import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from time import process_time

t1_start = process_time()

df_benign = pd.read_csv(r"E:\Doan\FinalDataset\URL\Benign_list_big_final.csv", header=None)
df_deface = pd.read_csv(r"E:\Doan\FinalDataset\URL\DefacementSitesURLFiltered.csv", header=None)
df_malware = pd.read_csv(r"E:\Doan\FinalDataset\URL\Malware_dataset.csv", header=None)
df_phishing = pd.read_csv(r"E:\Doan\FinalDataset\URL\phishing_dataset.csv", header=None)
df_spam = pd.read_csv(r"E:\Doan\FinalDataset\URL\spam_dataset.csv", header=None)
#df_add = pd.read_csv(r'E:\Datasets_for_ML\Dataset\Doan\Using-machine-learning-to-detect-malicious-URLs-master\data\data.csv')
df_benign = df_benign.drop_duplicates();
df_spam = df_spam.drop_duplicates();
df_malware = df_malware.drop_duplicates();
df_deface = df_deface.drop_duplicates();
df_phishing = df_phishing.drop_duplicates();

df_benign['label'] = 'benign'
df_deface['label'] = 'deface'
df_spam['label'] = 'spam'
df_malware['label'] = 'malware'
df_phishing['label'] = 'phishing'
#df_add['label'] = 0

#print(df_add)

df_benign.columns = ['URL', 'label']
df_deface.columns = ['URL', 'label']
df_spam.columns = ['URL', 'label']
df_malware.columns = ['URL', 'label']
df_phishing.columns = ['URL', 'label']
#df_add.columns = ['URL', 'label']
#print(df_benign.shape)
#print(df_benign.head())

df = pd.concat([df_benign, df_malware])
df = pd.concat([df, df_spam])
df = pd.concat([df, df_deface])
df = pd.concat([df, df_phishing])
#df = pd.concat([df, df_add])

print('-----------1.DONE--------------')
def makeTokens(f):
    token_By_Slash = str(f.encode('utf-8')).split('/')	#loai bo dau /
    total_Tokens = []
    for i in token_By_Slash:
        token_By_Dash = str(i).split('-')	#loai bo dau -
        token_By_Dot = []
        for j in range(0,len(token_By_Dash)):
            temp_Tokens = str(token_By_Dash[j]).split('.')	#loai bo dau .
            token_By_Dot = token_By_Dot + temp_Tokens
        total_Tokens = total_Tokens + token_By_Dash + token_By_Dot
    total_Tokens = list(set(total_Tokens))	#loai bo du thua tren
    return total_Tokens

#print(df.shape)
#print(df.head())
data = df['URL']
Label_binarizer = LabelBinarizer()
y = Label_binarizer.fit_transform(df['label']) #One hot encoder

print(y)
print(y.shape)
print('-----------2.DONE--------------')

#vectorizer = TfidfVectorizer(tokenizer=makeTokens, max_features=10000) # su dung custom Tokenizer
vectorizer = TfidfVectorizer(min_df=0.0, analyzer='char', sublinear_tf=True, ngram_range=(3, 3), tokenizer=makeTokens)
X = vectorizer.fit_transform(data) # Luu vao vector X sau khi TfIdf
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print('-----------3.DONE-------------')
print(y_test)
print(y_test.shape)

model = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

print('------------ALLDONE-------------')

t1_stop = process_time()
print("Elapsed time during the whole program in seconds:",t1_stop-t1_start)
