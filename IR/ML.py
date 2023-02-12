import pandas as pd
import numpy as np

df = pd.read_csv("sc.csv", header=None)
datas = df.values.tolist()

X = np.array([d[:3] for d in datas])
Y = np.array([d[-1] for d in datas])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=0.7, random_state=100)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
def get_socre(model, method):
    model.fit(x_train, y_train)
    result = model.predict(X)
    acc = accuracy_score(Y, result)
    f1 = f1_score(Y, result)
    p = precision_score(Y, result)
    r = recall_score(Y, result)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y, result)
    # roc_auc = auc(false_positive_rate, true_positive_rate)
    # roc_auc = round(roc_auc*100,3)
    acc = round(acc*100,2)
    f1 = round(f1*100,2)
    p = round(p*100,2)
    r = round(r*100,2)
    print(method,' Acc:', acc, ' Precision:', p, ' Recall:', r, r' F1:', f1, )

count = 0
for i in y_train:
    if(i==1):
        count+=1
print(count)
print(len(y_train)-count)


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
get_socre(clf, 'KNN')
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=0.01)
get_socre(clf, 'NB')
from sklearn.svm import SVC
clf = SVC(kernel='rbf', probability=True)
get_socre(clf, 'SVM')
from sklearn import tree
model_dt = tree.DecisionTreeClassifier()
get_socre(model_dt, 'DT')
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier()
get_socre(clf_rf, 'RF')
# import joblib
# joblib.dump(clf_rf, '../model/SCSD/sc_decide.model')