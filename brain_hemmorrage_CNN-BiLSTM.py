import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, roc_auc_score

# ----------------------------
# Parameters
# ----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
DATASET_DIR = './dataset/content/drive/MyDrive/Brain/Data'  

# ----------------------------
# Data Augmentation & Generators
# ----------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.1
)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# ----------------------------
# Build Hybrid CNN-BiLSTM Model
# ----------------------------
model = models.Sequential(name='Hybrid_CNN_BiLSTM')

# CNN block
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())

# Flatten and reshape for BiLSTM
model.add(layers.Reshape((-1, 128)))

# BiLSTM layer
model.add(layers.Bidirectional(layers.LSTM(64)))

# Dropout & Dense layers
model.add(layers.Dropout(DROPOUT_RATE))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# ----------------------------
# Compile Model
# ----------------------------
optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# ----------------------------
# Train Model
# ----------------------------
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# ----------------------------
# Evaluate Model
# ----------------------------
val_generator.reset()
preds = model.predict(val_generator, verbose=1)
pred_labels = (preds > 0.5).astype(int)
true_labels = val_generator.classes

print("\nClassification Report:")
print(classification_report(true_labels, pred_labels))

roc_auc = roc_auc_score(true_labels, preds)
print(f"Validation ROC-AUC: {roc_auc:.4f}")
