import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pickle
import os

# Parameters
im_size = 224
batch_size = 64
epochs = 20

# Load labels
labels_df = pd.read_csv("data/labels.csv")
labels_df['img_file'] = labels_df['id'].apply(lambda x: x + ".jpg")

# Initialize YOLO model
yolo_model = YOLO('yolov8n.pt')

# Function to crop dog from image using YOLO
def crop_dog_from_image(image_path, yolo_model):
    results = yolo_model(image_path)
    detections = results[0].boxes.xyxy.cpu().numpy()

    if len(detections) == 0:
        print(f"No dog detected in {image_path}.")
        return None

    x1, y1, x2, y2 = map(int, detections[0][:4])
    image = Image.open(image_path)
    cropped = image.crop((x1, y1, x2, y2))
    return cropped

# Preprocess images
def preprocess_dataset(df, base_dir):
    X, y = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(base_dir, row['img_file'])
        cropped_img = crop_dog_from_image(img_path, yolo_model)
        if cropped_img is None:
            continue

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

# Train/test split
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

# Model setup
resnet = ResNet50V2(input_shape=[im_size, im_size, 3], weights='imagenet', include_top=False)
for layer in resnet.layers[-50:]:
    layer.trainable = False

x = resnet.output
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_breeds, activation='softmax')(x)

model = Model(inputs=resnet.input, outputs=predictions)

optimizer = Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=["accuracy"])

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model and save history
history = model.fit(
    train_generator,
    steps_per_epoch=len(x_train) // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=len(x_test) // batch_size,
    callbacks=[reduce_lr, early_stop]
)

# Save model, encoder, and training history
model.save("model")
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
with open("train_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("Training complete and model saved.")

# Reload history for plots
with open("train_history.pkl", "rb") as f:
    history_data = pickle.load(f)

# Plot accuracy
plt.figure(figsize=(10, 5))
plt.plot(history_data['accuracy'], label='Train Accuracy')
plt.plot(history_data['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('metrics_accuracy.png')
plt.close()

# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(history_data['loss'], label='Train Loss')
plt.plot(history_data['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig('metrics_loss.png')
plt.close()

# Evaluate model
y_prob = model.predict(x_test, batch_size=32, verbose=1)
y_pred = np.argmax(y_prob, axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(15, 15))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('metrics_confusion_matrix.png')
plt.close()

# Accuracy metrics
rep = classification_report(y_test, y_pred, target_names=encoder.classes_, digits=3)
top1 = accuracy_score(y_test, y_pred)
acc_summary = f"\nTop-1 accuracy : {top1:.4f}\n"

# Top-k accuracy
def top_k_accuracy(y_true, y_prob, k=5):
    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
    return np.mean([y_true[i] in top_k_preds[i] for i in range(len(y_true))])

top3 = top_k_accuracy(y_test, y_prob, k=3)
top5 = top_k_accuracy(y_test, y_prob, k=5)
topk_summary = f"Top-3 accuracy: {top3:.4f}\nTop-5 accuracy: {top5:.4f}\n"

# Save everything
print(rep)
print(acc_summary)
print(topk_summary)

with open("metrics.txt", "w") as f:
    f.write(rep)
    f.write(acc_summary)
    f.write(topk_summary)
