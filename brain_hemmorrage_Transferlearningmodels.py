# transfer_learning_models.py

import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, roc_auc_score

# ----------------------------
# Parameters
# ----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
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
# Function to build transfer learning model
# ----------------------------
def build_transfer_model(base_model, model_name):
    base_model.trainable = False  # Freeze convolutional layers

    model = models.Sequential(name=model_name)
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))

    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# ----------------------------
# Build Models
# ----------------------------
vgg_base = applications.VGG16(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
resnet_base = applications.ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
densenet_base = applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))

vgg_model = build_transfer_model(vgg_base, 'VGG16_Transfer')
resnet_model = build_transfer_model(resnet_base, 'ResNet50_Transfer')
densenet_model = build_transfer_model(densenet_base, 'DenseNet121_Transfer')

# ----------------------------
# Train & Evaluate Function
# ----------------------------
def train_and_evaluate(model, train_gen, val_gen):
    print(f"\nTraining model: {model.name}")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen
    )
    val_gen.reset()
    preds = model.predict(val_gen, verbose=1)
    pred_labels = (preds > 0.5).astype(int)
    true_labels = val_gen.classes

    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels))

    roc_auc = roc_auc_score(true_labels, preds)
    print(f"{model.name} Validation ROC-AUC: {roc_auc:.4f}")

# ----------------------------
# Run Training & Evaluation
# ----------------------------
train_and_evaluate(vgg_model, train_generator, val_generator)
train_and_evaluate(resnet_model, train_generator, val_generator)
train_and_evaluate(densenet_model, train_generator, val_generator)
