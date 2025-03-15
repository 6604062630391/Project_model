import os
import numpy as np
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_DIR1 = "dataset/ham10000_images_part_1"
IMG_DIR2 = "dataset/ham10000_images_part_2"
METADATA_PATH = "dataset/HAM10000_metadata.csv"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10


df = pd.read_csv(METADATA_PATH)


image_paths = {}
for dirname in [IMG_DIR1, IMG_DIR2]:
    for img in os.listdir(dirname):
        image_paths[img.split(".")[0]] = os.path.join(dirname, img)


df["image_path"] = df["image_id"].map(image_paths)

df = df.dropna(subset=["image_path"])

df["dx"] = df["dx"].astype(str)
class_labels = sorted(df["dx"].unique())
label_to_index = {label: idx for idx, label in enumerate(class_labels)}
df["label_idx"] = df["dx"].map(label_to_index)

np.save("class_labels.npy", class_labels)

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label_idx"], random_state=42)

#Data Augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,

    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col="image_path",
    y_col="dx",
    target_size=IMAGE_SIZE,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col="image_path",
    y_col="dx",
    target_size=IMAGE_SIZE,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Build CNN model (MobileNetV2)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
preds = Dense(len(class_labels), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=preds)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

checkpoint = ModelCheckpoint("best_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[checkpoint]
)
