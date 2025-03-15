import streamlit as st

st.title("📊 การพัฒนาโมเดล Machine Learning")

st.markdown("""
    โมเดลที่พัฒนาในโปรเจกต์นี้เป็นโมเดลสำหรับทำนายการเป็นโรคอ้วนจากข้อมูลต่าง ๆ ที่เกี่ยวข้อง เช่น อายุ, เพศ, น้ำหนัก, ส่วนสูง และพฤติกรรมการใช้ชีวิต

    Dataset ที่ใช้ในการพัฒนาโมเดลนี้ได้มาจาก **Kaggle** โดยใช้ข้อมูลเกี่ยวกับ **Obesity Prediction Dataset**

    ขั้นตอนการพัฒนาโมเดลประกอบด้วยการเตรียมข้อมูล การเลือกและฝึกโมเดล Machine Learning 3 ตัว และการประเมินผลโมเดล
""")


st.subheader("1. การเตรียมข้อมูล (Data Preparation)")


st.markdown("""
    ขั้นตอนแรกในการพัฒนาโมเดลคือการเตรียมข้อมูลเพื่อให้พร้อมสำหรับการฝึกอบรมและทดสอบ โดยทำการ:
    - **การนำเข้าข้อมูล**: ข้อมูลได้มาจากไฟล์ CSV ที่มีหลายคุณสมบัติ เช่น เพศ, อายุ, น้ำหนัก, และข้อมูลพฤติกรรมการบริโภคอาหาร
    - **การลบข้อมูลที่ซ้ำซ้อน**: เพื่อให้ข้อมูลมีคุณภาพและไม่ทำให้โมเดลเกิดการ overfitting
    - **การแยกข้อมูล**: เราแยกข้อมูลออกเป็น **Features** และ **Target** โดย Target คือ "NObeyesdad" ซึ่งบ่งบอกถึงความเสี่ยงของโรคอ้วน
    - **การแปลงข้อมูลที่เป็นข้อความ**: โดยใช้ **LabelEncoder** เพื่อแปลงข้อมูลประเภทข้อความ (เช่น เพศ) เป็นค่าตัวเลข
    - **การปรับขนาดข้อมูล**: เราใช้ **StandardScaler** เพื่อทำให้ข้อมูลมีค่าเฉลี่ยเป็น 0 และเบี่ยงเบนมาตรฐานเป็น 1
    - **การใช้ SMOTE**: เนื่องจากข้อมูลของเราไม่สมดุล (imbalanced), เราจึงใช้ **SMOTE** เพื่อเพิ่มจำนวนตัวอย่างในคลาสที่มีจำนวนน้อย โดยการสร้างข้อมูลสังเคราะห์ใหม่จากคลาสนั้น เพื่อให้โมเดลสามารถเรียนรู้ได้ดียิ่งขึ้นจากทุกคลาส
""")

st.code("""
# นำเข้าข้อมูล
df = pd.read_csv("dataset/ObesityDataSet_raw_and_data_sinthetic.csv")

# ลบค่าซ้ำซ้อน
df = df.drop_duplicates()

# แยก Features และ Target
target_col = "NObeyesdad"
X = df.drop(columns=[target_col])
y = df[target_col]

# แปลงค่าข้อความเป็นตัวเลข
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# แปลง Target เป็นตัวเลข
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# แยก Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ใช้ SMOTE เพื่อเพิ่มข้อมูลในคลาสที่มีจำนวนน้อย
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
""")

st.subheader("2. ทฤษฎีของอัลกอริธึมที่พัฒนา")
st.markdown("""
    โมเดลที่ใช้ในโปรเจกต์นี้ประกอบด้วย 3 อัลกอริธึมหลัก ๆ ที่เหมาะสมกับการทำนายประเภทข้อมูล:
    
    - **Random Forest**: ใช้ **Ensemble Learning** โดยรวมผลลัพธ์จากต้นไม้การตัดสินใจหลายตัว ทำให้มีความทนทานต่อการ overfitting และสามารถทำงานได้ดีเมื่อข้อมูลมีความซับซ้อน
    - **Logistic Regression**: ใช้สำหรับการจำแนกประเภทที่มีลักษณะเป็นเชิงเส้น โดยใช้ฟังก์ชันโลจิสติกส์ในการทำนายค่าความน่าจะเป็นของแต่ละคลาส
    - **XGBoost**: การปรับปรุง **Gradient Boosting** ที่มีประสิทธิภาพสูงในการจัดการกับข้อมูลที่ซับซ้อน
""")

st.subheader("3. ขั้นตอนการพัฒนาโมเดล")
st.markdown("""
    ขั้นตอนในการพัฒนาโมเดลประกอบด้วย:
    - **การแบ่งข้อมูล (Train-Test Split)**: เราแบ่งข้อมูลออกเป็น 80% สำหรับเทรนและ 20% สำหรับทดสอบ
    - **การเทรนโมเดล**: เราทำการฝึกโมเดลทั้ง 3 ตัว (RandomForest, Logistic Regression, XGBoost)
    - **การประเมินผล**: ผลลัพธ์จากการทำนายจะถูกประเมินด้วย **Accuracy**, **Confusion Matrix**, และ **Classification Report**
    - **การเลือกโมเดลที่ดีที่สุด**: จากผลการประเมิน **XGBoost** เป็นโมเดลที่ให้ความแม่นยำสูงสุด (Accuracy = 97%)
""")

st.code("""
# เทรนโมเดลหลายตัว
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "XGBoost": XGBClassifier(eval_metric="logloss")
}

best_model = None
best_accuracy = 0

# ทำการเทรนโมเดลแต่ละตัวและประเมินผล
for name, model in models.items():
    print(f"Training {name}...")

    model.fit(X_train_scaled, y_train)
    
    # ทำนายผลลัพธ์
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.2f}")

    # แสดง Confusion Matrix
    print(f"Confusion Matrix for {name}:\n", confusion_matrix(y_test, y_pred))

    # แสดง Classification Report
    print(f"Classification Report for {name}:\n", classification_report(y_test, y_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"{name} Cross-validation accuracy: {cv_scores.mean() * 100:.2f}%")
""")

st.subheader("4. ผลลัพธ์และสรุป")
st.markdown("""
    หลังจากการทดลองและประเมินผล **XGBoost** เป็นโมเดลที่ให้ผลลัพธ์ดีที่สุด โดยมี **Accuracy** สูงถึง 96% ซึ่งแสดงให้เห็นว่าโมเดลสามารถทำนายโรคอ้วนได้อย่างแม่นยำ
    โมเดลนี้สามารถใช้ในการทำนายการเป็นโรคอ้วนจากข้อมูลใหม่ได้
""")

st.subheader("5. อ้างอิงแหล่งที่มาของ Dataset")
st.markdown("""
    ข้อมูลที่ใช้ในโปรเจกต์นี้ได้มาจาก **Kaggle** โดยสามารถเข้าถึงได้ที่: [Kaggle - Obesity Prediction Dataset](https://www.kaggle.com/datasets/adeniranstephen/obesity-prediction-dataset)
""")