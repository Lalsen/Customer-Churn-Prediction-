import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Clean and preprocess
df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Label encode binary categorical columns
le = LabelEncoder()
binary_cols = [col for col in df.columns if df[col].nunique() == 2]
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# One-hot encode multi-class categorical columns
multi_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() > 2]
df = pd.get_dummies(df, columns=multi_cols)

# Correlation matrix (optional visualization)
plt.figure(figsize=(14, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix[['Churn']].sort_values(by='Churn', ascending=False), annot=True, cmap='coolwarm')
plt.title("Correlation of Features with Churn")
plt.tight_layout()
plt.show()

# Features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# SMOTE to balance classes
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# ===============================
# 1️⃣ XGBoost
# ===============================
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    colsample_bytree=1.0,
    learning_rate=0.01,
    max_depth=8,
    n_estimators=200,
    subsample=0.8,
    random_state=42
)
xgb_model.fit(X_train_bal, y_train_bal)
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("------ XGBOOST ------")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Precision:", precision_score(y_test, y_pred_xgb))
print("Recall:", recall_score(y_test, y_pred_xgb))
print("F1 Score:", f1_score(y_test, y_pred_xgb))
print("ROC AUC:", roc_auc_score(y_test, y_prob_xgb))
print(classification_report(y_test, y_pred_xgb))

# ===============================
# 2️⃣ CatBoost
# ===============================
cat_model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.01,
    depth=8,
    verbose=0,
    random_state=42
)
cat_model.fit(X_train_bal, y_train_bal)
y_pred_cat = cat_model.predict(X_test)
y_prob_cat = cat_model.predict_proba(X_test)[:, 1]

print("------ CATBOOST ------")
print("Accuracy:", accuracy_score(y_test, y_pred_cat))
print("Precision:", precision_score(y_test, y_pred_cat))
print("Recall:", recall_score(y_test, y_pred_cat))
print("F1 Score:", f1_score(y_test, y_pred_cat))
print("ROC AUC:", roc_auc_score(y_test, y_prob_cat))
print(classification_report(y_test, y_pred_cat))

# ===============================
# 3️⃣ LightGBM
# ===============================
lgb_model = LGBMClassifier(
    learning_rate=0.01,
    max_depth=8,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=1.0,
    random_state=42
)
lgb_model.fit(X_train_bal, y_train_bal)
y_pred_lgb = lgb_model.predict(X_test)
y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]

print("------ LIGHTGBM ------")
print("Accuracy:", accuracy_score(y_test, y_pred_lgb))
print("Precision:", precision_score(y_test, y_pred_lgb))
print("Recall:", recall_score(y_test, y_pred_lgb))
print("F1 Score:", f1_score(y_test, y_pred_lgb))
print("ROC AUC:", roc_auc_score(y_test, y_prob_lgb))
print(classification_report(y_test, y_pred_lgb))
