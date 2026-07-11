import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans


df = pd.read_csv("churn.csv")


df.drop("customerID", axis=1, inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = pd.get_dummies(df)

X = df.drop("Churn", axis=1)
y = df["Churn"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# Decision Tree

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("===== Decision Tree =====")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, dt_pred))
print("Classification Report:")
print(classification_report(y_test, dt_pred))



#Feature Importance Analysis

print("\n===== Feature Importance (Decision Tree) =====")
importances = dt_model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("Top 15 important features:")
print(feature_importance_df.head(15).to_string(index=False))

print("\nImportant features above 0.01:")
for feature, importance in zip(X.columns, importances):
    if importance > 0.01:
        print(feature, ":", importance)


# SVM
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model = SVC(kernel="linear", random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)

print("\n===== SVM =====")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, svm_pred))
print("Classification Report:")
print(classification_report(y_test, svm_pred))




print("\n===== K-Means Clustering =====")

kmeans = KMeans(n_clusters=2, random_state=42)

clusters = kmeans.fit_predict(X)

df["Cluster"] = clusters

print("\nCluster counts:")
print(df["Cluster"].value_counts())

print("\nCluster Analysis (Mean values):")
print(df.groupby("Cluster").mean())