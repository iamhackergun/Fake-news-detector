import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("test.csv")
df.head(10)

x = df["Statement"]
y = df["Label"]
x_train, x_test,y_train, y_test = train_test_split(x, y, test_size=10)

#vectorizing text
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

#models
#logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(xv_train, y_train)
#accuracy
lr.score(xv_test, y_test)
pred_lr = lr.predict(xv_test)
#print(classification_report(y_test, pred_lr))

#decision tree classifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(xv_train, y_train)
#accuracy
dt.score(xv_test, y_test)
pred_dt = dt.predict(xv_test)
#print(classification_report(y_test,pred_dt))

#gradient boosting classifier
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(random_state=0)
gbc.fit(xv_train, y_train)
#accuracy
gbc.score(xv_test, y_test)
pred_gbc = gbc.predict(xv_test)
#print(classification_report(y_test,pred_gbc))

#random forest classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=0)
rf.fit(xv_train, y_train)
#accuracy
rf.score(xv_test,y_test)
pred_rf = rf.predict(xv_test)
#print(classification_report(y_test,pred_rf)

#manual testing
def output_lable(n):
    if n == 0:
        return "fake news"
    elif n == 1:
        return "not a fake news"

def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_lr = lr.predict(new_xv_test)
    pred_dt = dt.predict(new_xv_test)
    pred_gbc = gbc.predict(new_xv_test)
    pred_rf = rf.predict(new_xv_test)

    return print("\nprediction: {}".format(output_lable(pred_dt)))

news = str(input())
manual_testing(news)