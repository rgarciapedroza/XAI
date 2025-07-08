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

data = pd.read_csv("alzheimers_disease_data.csv")
data.drop(columns=["PatientID","DoctorInCharge"], inplace=True)
pd.set_option('display.max_columns', None)

data = data.dropna()

label_encoders = {}
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.drop("Diagnosis", axis=1)
y = data["Diagnosis"]

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

def plot_lime(model, instance_index=5, name="Model"):
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X.columns.tolist(),
        class_names=['No Alzheimer', 'Alzheimer'],  # Adaptado al problema de diagnóstico
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