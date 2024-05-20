import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm


def Create_Directory_DataFrame():
    df = pd.DataFrame(columns=['Class', 'Location'])
    basedir = './Datasets/leapGestRecog/'

    for folder in os.listdir(basedir):
        for Class in os.listdir(os.path.join(basedir, folder)):
            for location in os.listdir(os.path.join(basedir, folder, Class)):
                df = pd.concat([df, pd.DataFrame(
                    {'Class': [Class], 'Location': [os.path.join(basedir, folder, Class, location)]})],
                               ignore_index=True)

    df = df.sample(frac=1).reset_index(drop=True)
    return df


def conv_block(filters):
    block = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D()
    ])
    return block


def dense_block(units, dropout_rate):
    block = tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate)
    ])
    return block


def build_model(act, final_class, w, h):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(w, h, 1)),

        tf.keras.layers.Conv2D(16, 3, activation=act, padding='same'),
        tf.keras.layers.Conv2D(16, 3, activation=act, padding='same'),
        tf.keras.layers.MaxPool2D(),

        conv_block(32),
        conv_block(64),

        conv_block(128),
        tf.keras.layers.Dropout(0.3),  # Increased dropout rate

        conv_block(256),
        tf.keras.layers.Dropout(0.3),  # Increased dropout rate

        tf.keras.layers.Flatten(),
        dense_block(512, 0.5),  # Increased dropout rate
        dense_block(128, 0.4),  # Increased dropout rate
        dense_block(64, 0.3),

        tf.keras.layers.Dense(final_class, activation='sigmoid')
    ])
    return model


dataframe = Create_Directory_DataFrame()
w, h = 64, 64
final_class = 10

# Read and preprocess images
train_image = [cv2.resize(cv2.imread(location, 0), (w, h), interpolation=cv2.INTER_AREA).reshape(w, h, 1) / 255.0
               for location in tqdm(dataframe['Location'])]

# Convert the list to a NumPy array
X = np.array(train_image)
y = dataframe['Class'].values.reshape(-1, 1)
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(y)
y = enc.transform(y).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

model = build_model('relu', final_class, w, h)
METRICS = [
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
]
model.compile(
                optimizer='RMSprop',
                loss='categorical_crossentropy',
                metrics=METRICS
        )
history = model.fit(X_train, y_train, epochs=50, validation_split=0.3, batch_size=15,verbose=1,shuffle=True)
y_pred = model.evaluate(X_test , y_test,verbose =1)
y_prediction = model.predict(X_test)