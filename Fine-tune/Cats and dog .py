#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import tensorflow_datasets as tfds

# Load the dataset
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

# Resize images to the expected ResNet input size
IMG_SIZE = 224
def format_example(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

# Shuffle and batch the data
BATCH_SIZE = 32
train_batches = train.shuffle(1000).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation.batch(BATCH_SIZE).prefetch(1)
test_batches = test.batch(BATCH_SIZE)

# Create the base model from pre-trained ResNet50
base_model = ResNet50(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                      include_top=False,
                      weights='imagenet')

# Freeze the convolutional base
base_model.trainable = False

# Add custom layers
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
EPOCHS = 5
history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)

# Evaluate the model
loss, accuracy = model.evaluate(test_batches)
print("\nTest accuracy: {:.2f}%".format(accuracy * 100))

# Save the model (optional)
# model.save("cats_vs_dogs_resnet50.h5")



