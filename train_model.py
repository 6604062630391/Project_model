from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
from collections import Counter

df = pd.read_csv("dataset/ObesityDataSet_raw_and_data_sinthetic.csv")

# ลบค่าซ้ำซ้อน
df = df.drop_duplicates()

# แยก Features และ Target
target_col = "NObeyesdad"
X = df.drop(columns=["TUE",target_col])
y = df[target_col]

label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

le_target = LabelEncoder()
y = le_target.fit_transform(y)

# แยก Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ก่อนการทำ SMOTE ตรวจสอบการกระจายของข้อมูล
print(f"Before SMOTE: {Counter(y_train)}")

# ใช้ SMOTE เพื่อเพิ่มจำนวนข้อมูลในคลาสที่มีน้อย
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# ตรวจสอบการกระจายของข้อมูลหลังจากการใช้ SMOTE
print(f"After SMOTE: {Counter(y_train_smote)}")

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        eval_metric="logloss"
    )
}

best_model = None
best_accuracy = 0

# ทำการเทรนโมเดลแต่ละตัวและประเมินผล
for name, model in models.items():
    print(f"Training {name}...")

    # ใช้ข้อมูลที่ผ่านการทำ SMOTE สำหรับการฝึก
    model.fit(X_train_smote, y_train_smote)
    
    # ทำนายผลลัพธ์
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.2f}")

    # แสดง Confusion Matrix
    print(f"Confusion Matrix for {name}:\n", confusion_matrix(y_test, y_pred))

    # แสดง Classification Report
    print(f"Classification Report for {name}:\n", classification_report(y_test, y_pred))
    
    joblib.dump(model, f"models/{name}_model.pkl")

    # การเลือกโมเดลที่ดีที่สุด
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = name

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5)
    print(f"{name} Cross-validation accuracy: {cv_scores.mean() * 100:.2f}%")
    y_train_pred = model.predict(X_train_smote)
    train_acc = accuracy_score(y_train_smote, y_train_pred)
    print(f"{name} Training Accuracy: {train_acc:.2f}")

    

joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")
joblib.dump(le_target, "models/target_encoder.pkl")

print(f"✅ Training Complete! Best Model: {best_model} with Accuracy: {best_accuracy:.2f}")


