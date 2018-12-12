
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


#Create your df here:
df = pd.read_csv("profiles.csv")

#Data Exploration
df.head()
df.height.head(n = 20)
df.height.isnull().values.any()
df.height.describe()
df.body_type.head(n=20)
df.body_type.isnull().values.any()
df.body_type.unique()

#Data visualization
plt.hist(df.height[df['height'].notnull()], bins=25)
plt.xlabel('Height')
plt.ylabel('Frequency')
plt.title('Height Histogram')
plt.xlim(1., 95.)
plt.show()


df.groupby(['body_type','sex'])['essay0'].size().unstack().plot(kind='bar',stacked=True)
plt.xlabel('Body Type')
plt.ylabel('Frequency')
plt.title('Frequency of Body Type by Gender')
plt.show()


drinks_status = df.drinks.unique()
from collections import Counter
status_counts = Counter(df.drinks)
fig1, ax1 = plt.subplots(figsize = (12,6), subplot_kw=dict(aspect="equal"))
counts = [float(v) for v in status_counts.values()]
statuses = [k for k in status_counts]
wedges, texts, autotexts = ax1.pie(counts, autopct='%1.1f%%',
                                  textprops=dict(color="w"))
ax1.legend(wedges, statuses,
          title="Status",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=10, weight="bold")
ax1.set_title("Breakdown by Drinking Habits")
plt.show()

drugs_status = df.drugs.unique()
from collections import Counter
status_counts = Counter(df.drugs)
fig1, ax1 = plt.subplots(figsize = (12,6), subplot_kw=dict(aspect="equal"))
counts = [float(v) for v in status_counts.values()]
statuses = [k for k in status_counts]
wedges, texts, autotexts = ax1.pie(counts, autopct='%1.1f%%',
                                  textprops=dict(color="w"))
ax1.legend(wedges, statuses,
          title="Status",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))
plt.setp(autotexts, size=10, weight="bold")
ax1.set_title("Breakdown by Drug Habits")
plt.show()


#Question: Can one predict body_type using diet, drinks, drugs, height, and age?


#Data augmentation
body_type_mapping = {'rather not say':0, 'overweight':1, 'full figured':2, 'a little extra':3, 'curvy':4, 'average':5, 
                     'thin':6, 'skinny':7, 'athletic':8, 'fit':9, 'jacked':10,'used up':11}
df["body_type_code"] = df.body_type.map(body_type_mapping)

df.diet.unique()
diet_mapping = {'strictly anything':0, 'mostly other':1, 'anything':2, 'vegetarian':3,'mostly anything':4, 
                'mostly vegetarian':5, 'strictly vegan':6,'strictly vegetarian':7, 'mostly vegan':8, 'strictly other':9,
                'mostly halal':10, 'other':11, 'vegan':12, 'mostly kosher':13, 'strictly halal':14,'halal':15, 
                'strictly kosher':16, 'kosher':17}
df["diet_code"] = df.diet.map(diet_mapping)

drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
df["drinks_code"] = df.drinks.map(drink_mapping)

df.drugs.unique()
drug_mapping = {"never": 0, "sometimes": 1, "often": 2}
df["drug_code"] = df.drugs.map(drug_mapping)

df.sex.unique()
sex_mapping = {"m": 0, "f": 1}
df["sex_code"] = df.sex.map(sex_mapping)


#Data Normalization

question_data = df.dropna(subset = ['body_type_code', 'diet_code', 'drinks_code', 'drug_code', 'sex_code', 'height'])
question_data = question_data[['body_type_code', 'diet_code', 'drinks_code', 'drug_code', 'sex_code', 'height']]

x = question_data.values
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)


question_data = pd.DataFrame(x_scaled, columns=question_data.columns)

question_data.head()


#Classification Techniques
#K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

#Create test and training groups
from sklearn.model_selection import train_test_split

question_data_variables = question_data[['diet_code', 'drinks_code', 'drug_code', 'sex_code', 'height']]
question_data_labels = question_data['body_type_code']

training_dataset, test_dataset, training_labels, test_labels = train_test_split(question_data_variables, question_data_labels, train_size =0.8, test_size = 0.2, random_state = 6)

lab_enc = LabelEncoder()
training_labels_encoded = lab_enc.fit_transform(training_labels)
test_labels_encoded = lab_enc.fit_transform(test_labels)

KNNclassifier = KNeighborsClassifier(146)
KNNclassifier.fit(training_dataset, training_labels_encoded)
knn_predict = KNNclassifier.predict(test_dataset)
print(KNNclassifier.score(test_dataset, test_labels_encoded))

accuracies = []
for k in range(1, 200):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_dataset, training_labels_encoded)
  accuracies.append(classifier.score(test_dataset, test_labels_encoded))

k_list = range(1, 200)
plt.plot(k_list, accuracies)
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.title('Body Type Classifier Accuracy')
plt.show()

error = []
for i in range(1, 200):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(training_dataset, training_labels_encoded)
    pred_i = knn.predict(test_dataset)
    error.append(np.mean(pred_i != test_labels_encoded))

plt.figure(figsize=(12, 6))  
plt.plot(range(1, 200), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')  
plt.show()


#SVMs
from sklearn.svm import SVC

SVCclassifier = SVC(kernel = 'rbf', gamma = 19)
SVCclassifier.fit(training_dataset, training_labels_encoded)
svc_predict = SVCclassifier.predict(test_dataset)
test_labels_encoded = lab_enc.fit_transform(test_labels)
print(svc_predict)
print(SVCclassifier.score(test_dataset, test_labels_encoded))

SVCclassifier = SVC(kernel = 'rbf', gamma = 20)
SVCclassifier.fit(training_dataset, training_labels_encoded)
svc_predict = SVCclassifier.predict(test_dataset)
test_labels_encoded = lab_enc.fit_transform(test_labels)
print(svc_predict)
print(SVCclassifier.score(test_dataset, test_labels_encoded))

SVCclassifier = SVC(kernel = 'rbf', gamma = 21)
SVCclassifier.fit(training_dataset, training_labels_encoded)
svc_predict = SVCclassifier.predict(test_dataset)
test_labels_encoded = lab_enc.fit_transform(test_labels)
print(svc_predict)
print(SVCclassifier.score(test_dataset, test_labels_encoded))


#Regression Techniques
#Multiple Linear Regression
from sklearn.linear_model import LinearRegression

mlr = LinearRegression()
mlr.fit(training_dataset, training_labels_encoded)
mlr_predict = mlr.predict(test_dataset)
print(mlr_predict)
print(mlr.score(test_dataset, test_labels_encoded))

plt.scatter(mlr_predict,test_labels)
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.show()


#KNN Regression
from sklearn.neighbors import KNeighborsRegressor

regressor = KNeighborsRegressor(n_neighbors = 146)
regressor.fit(training_dataset, training_labels)
KNNRegression_predict = regressor.predict(test_dataset)
KNNRegression_predict_labels_encoded = lab_enc.fit_transform(KNNRegression_predict)
print(regressor.score(test_dataset, test_labels))

plt.scatter(KNNRegression_predict,test_labels)
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.show()


#Analyze accuracy, precision, and recall
from sklearn.metrics import classification_report, confusion_matrix 
 
KNN_predict_labels_encoded = lab_enc.fit_transform(knn_predict)    
    
print(confusion_matrix(test_labels_encoded, KNN_predict_labels_encoded))  
print(classification_report(test_labels_encoded, KNN_predict_labels_encoded))  

svc_predict_labels_encoded = lab_enc.fit_transform(svc_predict) 

print(confusion_matrix(test_labels_encoded, svc_predict_labels_encoded))  
print(classification_report(test_labels_encoded, svc_predict_labels_encoded)) 

