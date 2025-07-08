import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import shap
import lime.lime_tabular
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

data = pd.read_csv("adult.csv")
data = data.dropna()
data = data.drop(columns=["fnlwgt", "education.num", "capital.gain", "capital.loss"])

label_encoders = {}
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.drop("income", axis=1)
y = data["income"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=21)

def evaluate_model(model, name="Model"):
    y_pred = model.predict(X_test)
    print(f"\n{name} - Classification Report:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"{name} - Confusion Matrix")
    plt.show()
    return y_pred

def plot_shap(model, name="Model"):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, show=True)

def plot_lime(model, instance_index=75, name="Model"):
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X.columns.tolist(),
        class_names=['<=50K', '>50K'],
        mode='classification',
        discretize_continuous=True
    )
    instance = X_test.iloc[instance_index]
    exp = explainer_lime.explain_instance(
        data_row=instance,
        predict_fn=lambda x: model.predict_proba(pd.DataFrame(x, columns=X.columns))
    )
    features, values = zip(*exp.as_list())
    colors = ['green' if v > 0 else 'red' for v in values]
    plt.figure(figsize=(8, 5))
    plt.barh(features, values, color=colors)
    plt.xlabel("LIME Weight")
    plt.title(f"LIME Explanation for Instance {instance_index} - {name}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def demographic_parity(y_pred, sensitive_feature):
    df = pd.DataFrame({'y_pred': y_pred, 'group': sensitive_feature})
    return df.groupby('group')['y_pred'].mean()

print("=== Logistic Regression ===")
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear'],
    'class_weight': ['balanced']
}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=21), param_grid, cv=5)
grid_lr.fit(X_train, y_train)
model_lr = grid_lr.best_estimator_

y_pred_lr = evaluate_model(model_lr, name="Logistic Regression")
plot_shap(model_lr, name="Logistic Regression")
plot_lime(model_lr, name="Logistic Regression")
print("Demographic Parity (LR):\n", demographic_parity(y_pred_lr, X_test['sex']))

print("\n=== HistGradientBoostingClassifier ===")
model_hgb = HistGradientBoostingClassifier(random_state=21)
model_hgb.fit(X_train, y_train)

y_pred_hgb = evaluate_model(model_hgb, name="HistGradientBoosting")
plot_shap(model_hgb, name="HistGradientBoosting")
plot_lime(model_hgb, name="HistGradientBoosting")
print("\nDemographic Parity (HGB):\n", demographic_parity(y_pred_hgb, X_test['sex']))
