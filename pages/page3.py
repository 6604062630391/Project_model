import streamlit as st

st.title("🧠 การพัฒนาโมเดล Neural Network")

st.markdown("""
โมเดลที่พัฒนาในโปรเจกต์นี้เป็นการพัฒนาโมเดล Deep Learning ประเภท **Convolutional Neural Network (CNN)** เพื่อใช้ในการจำแนกประเภทของโรคผิวหนังจากภาพถ่ายทางการแพทย์
            
Dataset ที่ใช้ในการพัฒนาโมเดลนี้ได้มาจาก **Kaggle** โดยใช้ **HAM10000 Dataset** ซึ่งเป็นชุดข้อมูลที่มีภาพผิวหนังและคำอธิบายของโรค 7 ประเภท

""")

st.subheader("1. การเตรียมข้อมูล (Data Preparation)")
st.markdown("""
ขั้นตอนการเตรียมข้อมูลมีดังนี้:

- **โหลดภาพและข้อมูลคำอธิบาย (metadata)** จากไฟล์ `HAM10000_metadata.csv`
- **เชื่อมโยงข้อมูลภาพ** กับ metadata โดยใช้ชื่อไฟล์
- **จัดกลุ่ม label** ของโรคผิวหนัง (เช่น Melanoma, Nevus, etc.)
- **การแปลงภาพ (Preprocessing)**:
  - Resize ภาพให้มีขนาดเท่ากัน เช่น 64x64 px
  - Normalize ค่า pixel (แบ่งด้วย 255)
- **One-hot Encoding** ของ Label เพื่อใช้ใน multi-class classification
- **Train-Test Split**: แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ (เช่น 80% / 20%)
""")

st.subheader("2. ทฤษฎีของอัลกอริธึมที่พัฒนา")
st.markdown("""
**Convolutional Neural Network (CNN)** เป็นโครงข่ายประสาทเทียมที่มีความสามารถในการวิเคราะห์ข้อมูลเชิงภาพ โดยมีส่วนประกอบหลัก:
- **Convolutional Layer**: ตรวจจับคุณลักษณะต่าง ๆ จากภาพ เช่น ขอบ, ลวดลาย
- **Pooling Layer**: ลดขนาดข้อมูลเพื่อลดจำนวนพารามิเตอร์และป้องกัน overfitting
- **Fully Connected Layer**: ใช้ในการตัดสินผลลัพธ์การจำแนกประเภท

การออกแบบโมเดลที่ใช้ในโปรเจกต์นี้มีโครงสร้าง 3 ชั้นของ Convolutional Layer ตามด้วย Flatten และ Dense Layer
""")

st.subheader("3. ขั้นตอนการพัฒนาโมเดล")
st.markdown("""
- **การออกแบบสถาปัตยกรรม CNN** เช่น:
```python
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(7, activation='softmax'))
""")
st.markdown("""- **การ compile และฝึกโมเดล:**
```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)
""")

st.markdown("""- **การประเมินผลลัพธ์ของโมเดล:**
```python
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy: ", accuracy)
""")

st.markdown("""- **การวิเคราะห์ Performance:
ดูค่า Accuracy, Confusion Matrix, Classification Report
แสดง Class Probability Distribution:**
```python
st.subheader("📊 Class Probability Distribution")
fig, ax = plt.subplots()
ax.pie(prediction[0], labels=class_labels, autopct="%1.1f%%", startangle=90)
ax.axis("equal")
st.pyplot(fig)
""")

st.subheader("4. ผลลัพธ์และสรุป")
st.markdown(""" โมเดล CNN ที่พัฒนาให้ผลลัพธ์ที่แม่นยำในการจำแนกภาพโรคผิวหนัง โดยสามารถใช้ทำนายประเภทของโรคในภาพใหม่ได้อย่างมีประสิทธิภาพ

ผลลัพธ์ Accuracy อยู่ในช่วง 75-85% แล้วแต่จำนวน Epoch และการปรับแต่ง Hyperparameter

Accuracy (Train): 79.3% | Accuracy (Validation): 77.1%
            
Loss (Train): 0.5718 |  Loss (Validation): 0.6558

""")
st.subheader("5. อ้างอิงแหล่งที่มาของ Dataset")
st.markdown("""
    ข้อมูลที่ใช้ในโปรเจกต์นี้ได้มาจาก **Kaggle** โดยสามารถเข้าถึงได้ที่: [Kaggle - Skin Cancer MNIST: HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/data)
""")