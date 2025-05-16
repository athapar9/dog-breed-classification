import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from ultralytics import YOLO
import os

# Parameters
im_size = 224
batch_size = 64
epochs = 20
learning_rate = 1e-3

# load labels
labels_df = pd.read_csv("data/labels.csv")
labels_df['img_file'] = labels_df['id'].apply(lambda x: x + ".jpg")

# Initialize YOLO model for dog detection
yolo_model = YOLO('yolov8n.pt')

# use YOLO to detect dogs/ crop dog from image 
def crop_dog_from_image(image_path, yolo_model):
    results = yolo_model(image_path)
    detections = results[0].boxes.xyxy.cpu().numpy()

    if len(detections) == 0:
        print(f"No dog detected in {image_path}.")
        return None

    # Use first detected box
    x1, y1, x2, y2 = map(int, detections[0][:4])
    image = Image.open(image_path)
    cropped = image.crop((x1, y1, x2, y2))
    return cropped

# resizes and preprocess for ResNet
def preprocess_dataset(df, base_dir):
    X = []
    y = []
    for idx, row in df.iterrows():
        img_path = os.path.join(base_dir, row['img_file'])
        cropped_img = crop_dog_from_image(img_path, yolo_model)
        if cropped_img is None:
            continue  # skip images with no dog detected

        img_resized = cropped_img.resize((im_size, im_size))
        img_array = preprocess_input(np.array(img_resized).astype(np.float32))
        X.append(img_array)
        y.append(row['breed'])
    
    return np.array(X), np.array(y)

print("Preparing dataset...")
X, y = preprocess_dataset(labels_df, "data/train/")
print(f"Finished preparing dataset. Number of images: {len(X)}")

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_breeds = len(encoder.classes_)
print(f"Number of breeds: {num_breeds}")

# Split train/test data (80/20)
x_train, x_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
test_generator = test_datagen.flow(x_test, y_test, batch_size=batch_size)

# Build classifier model using ResNet50V2 
resnet = ResNet50V2(input_shape=[im_size, im_size, 3], weights='imagenet', include_top=False)
for layer in resnet.layers:
    layer.trainable = False

x = resnet.output
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_breeds, activation='softmax')(x)

model = Model(inputs=resnet.input, outputs=predictions)

optimizer = RMSprop(learning_rate=learning_rate, rho=0.9)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=["accuracy"])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=len(x_train) // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=len(x_test) // batch_size,
    callbacks=[reduce_lr, early_stop]
)

# Save model and encoder
model.save("model")
import pickle
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("Training complete and model saved.")