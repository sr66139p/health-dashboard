#Load Libraries
import pandas as pd
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
#import matplotlib.pyplot as plt 
#from sklearn.metrics import confusion_matrix

col_names = ['pH', 'pH Status', 'Temperature', 'Temperature Status', 'Salinity', 'Salinity Status', 'Turbidity', 'Turbidity Status', 'Dissolved Oxygen', 'Dissolved Oxygen Status']

#Loading Dataset
data = pd.read_csv("data_2.csv", header=None, names=col_names, index_col=None)

#data.to_csv('data_2.csv', header=None, index=False)
#df = data.dropna(subset = ["Rows"], axis = 1)

df = pd.DataFrame(data)
numpy_array = df.to_numpy()

#Dropping NaN Values
#df.dropna(subset = ["Rows"], axis = 0)

#df.head(5)

#Splitting data in features and target variable
#feature_cols = ['pH', 'pH Status', 'Temperature', 'Temperature Status', 'Salinity', 'Salinity Status', 'Turbidity', 'Turbidity Status', 'Dissolved Oxygen', 'Dissolved Oxygen Status']
feature_cols = ['pH', 'Temperature', 'Salinity', 'Turbidity', 'Dissolved Oxygen']
X = df[feature_cols] #Features
y= df[['pH Status', 'Temperature Status', 'Salinity Status', 'Turbidity Status', 'Dissolved Oxygen Status']] #Target Variable

for index, row in df.iterrows():
    print(f"Row {index}:")
    print(f"pH: {row['pH']}")
    print(f"pH Status: {row['pH Status']}")
    print(f"Temperature: {row['Temperature']}")
    print(f"Temperature Status: {row['Temperature Status']}")
    print(f"Salinity: {row['Salinity']}")
    print(f"Salinity Status: {row['Salinity Status']}")
    print(f"Turbidity: {row['Turbidity']}")
    print(f"Turbidity Status: {row['Turbidity Status']}")
    print(f"Dissolved Oxygen: {row['Dissolved Oxygen']}")
    print(f"Dissolved Oxygen Status: {row['Dissolved Oxygen Status']}")
    print("\n")

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Mapping for pH status labels
pH_status_mapping = {
    0: 'Bad', 1: 'Bad', 2: 'Bad', 3: 'Bad', 4: 'Bad',
    5: 'Good', 6: 'Good',
    7: 'Excellent', 8: 'Excellent',
    9: 'Warning', 10: 'Warning',
    11: 'Bad', 12: 'Bad', 13: 'Bad', 14: 'Bad'
}

# Assuming 'pH' is the name of your pH column, create a new 'pH Status' column
df['pH Status'] = df['pH'].map(pH_status_mapping)

#Converting pH column to float
#df['pH'] = df['pH'].astype(float)

# Create Decision Tree classifer object
#clf = DecisionTreeClassifier(max_leaf_nodes = 15)
#classifier = MultiOutputClassifier(clf, n_jobs=2)
#classifier = clf.fit(X_train, y_train)
#dlf = classifier.fit(X_train, y_train)

#Random Forest Classifier
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Train Decision Tree Classifer
#clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = rf.predict(X_test)

# Drop value column from the df y_test
# Convert the df to a numpy array (np.array)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error:v {mse}")

# Model Accuracy, how often is the classifier correct?#
# accuracy = accuracy_score(y_test, y_pred)

#print("Accuracy:", accuracy)
# Calculate Hamming Loss
#metric = hamming_loss(mode= 'multiclass', threshold = 0.6)

#hamming_loss_score = hamming_loss(y_true=y_test, y_pred=y_pred)

# Calculate Jaccard Score (average='samples' considers multiple labels)
#jaccard_score_score = jaccard_score(y_test, y_pred, average='samples')

#print("Jaccard Score:", jaccard_score_score)

'''def plot_confusion_matrix(y,y_pred):
    #"This function plots the confusion matrix"

    #cm = confusion_matrix(y, y_pred)
   # ax = plt.subplot()
   # sns.heatmap(cm, annot=True, ax = ax); 
    #ax.set_xlabel("Predicted Labels")
    #ax.set_ylabel("True Labels")
   # ax.set_title("Confusion Matrix");
   # ax.xaxis.set_ticklabels(['did no'])'''

