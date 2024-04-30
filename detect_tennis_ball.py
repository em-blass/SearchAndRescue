import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load and preprocess the data
train_dir = 'tennisdataset/train'
val_dir = 'tennisdataset/validation'
batch_size = 32
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(128, 128), batch_size=batch_size, class_mode='binary')
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(128, 128), batch_size=batch_size, class_mode='binary')

# Build model using MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model and evaluate on validation set
history = model.fit(
    train_generator, steps_per_epoch=len(train_generator), epochs=5,
    validation_data=val_generator, validation_steps=len(val_generator))

# Save the model to an H5 file
model.save('tennis_ball_detector_model.h5')

# Output the training and validation accuracy for each epoch
for i, (train_acc, val_acc) in enumerate(zip(history.history['accuracy'], history.history['val_accuracy'])):
    print(f"Epoch {i+1}: Training Acc = {train_acc:.4f}, Validation Acc = {val_acc:.4f}")

# Plot the training and validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png')  # Save the plot as an image file
plt.show()

# Generate predictions on the validation set
val_steps = len(val_generator)
val_generator.reset()
y_pred = model.predict(val_generator, steps=val_steps)
y_pred = np.round(y_pred)
y_true = val_generator.classes

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['No Tennis Ball', 'Tennis Ball'])
plt.yticks(tick_marks, ['No Tennis Ball', 'Tennis Ball'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')  # Save the plot as an image file
plt.show()

# Real-time detection with webcam (Adjust to Tello if necessary)
cap = cv2.VideoCapture(0)
threshold = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for prediction
    small_frame = cv2.resize(frame, (128, 128))
    small_frame = np.expand_dims(small_frame, axis=0)
    small_frame = small_frame / 255.0

    # Predict
    prediction = model.predict(small_frame)[0][0]
    print(f"Prediction score: {prediction}")

    if prediction > threshold:
        text = 'Tennis Ball Detected'
    else:
        text = 'No Tennis Ball'

    # Display the results
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Tennis Ball Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
