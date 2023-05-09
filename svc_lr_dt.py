# svc model 

df = pd.read_csv('UpdatedCompany.csv', on_bad_lines='skip')
X_train, X_test, y_train, y_test = train_test_split(df['Title'].values, df['Tag'].values, random_state=42)
cv = CountVectorizer()
model = Pipeline([('vect', cv), (
'clf', SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42, gamma=8))])
model.fit(X_train, y_train)
ypred = model.predict(X_test)
print(metrics.accuracy_score(y_test, ypred))
joblib.dump(model, 'CatModel')

# LogisticRegression and DecisionTreeClassifier

df = pd.read_csv("newaidos(wn).csv")
X_train,X_test,y_train,y_test = train_test_split(df.text,df.res,test_size=0.2,random_state=2)
cv = CountVectorizer()
nb_pipeline = Pipeline([('vect', cv), ('clf', MultinomialNB())])
nb_pipeline.fit(X_train, y_train)

clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf1.fit(X_train,y_train)
clf2.fit(X_train,y_train)
y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)

print("Accuracy of Logistic Regression",accuracy_score(y_test,y_pred1))
print("Accuracy of Decision Trees",accuracy_score(y_test,y_pred2))

confusion_matrix(y_test,y_pred1)
print("Logistic Regression Confusion Matrix\n")
pd.DataFrame(confusion_matrix(y_test,y_pred1),columns=list(range(0,2)))

print("Decision Tree Confusion Matrix\n")
pd.DataFrame(confusion_matrix(y_test,y_pred2),columns=list(range(0,2)))
result = pd.DataFrame()
result['Actual Label'] = y_test
result['Logistic Regression Prediction'] = y_pred1
result['Decision Tree Prediction'] = y_pred2

result.sample(10)

precision_score(y_test,y_pred1,average=None)

precision_score(y_test,y_pred2,average=None)

recall_score(y_test,y_pred2,average=None)