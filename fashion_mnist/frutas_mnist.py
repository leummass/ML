import os
from tensorflow import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

num_classes = 4
def cargar_imagenes_y_etiquetas(carpeta, etiqueta):
    datos = []
    for imagen_nombre in os.listdir(carpeta):
        imagen_ruta = os.path.join(carpeta, imagen_nombre)
        imagen = Image.open(imagen_ruta).resize((100, 100))  # Cambia el tamaño de las imágenes según sea necesario
        imagen_array = np.array(imagen)
        datos.append([imagen_array, etiqueta])
    return datos
carpeta_entrenamiento = "C:\entrenamiento"
carpeta_prueba = "C:\prueba"
etiquetas = ["aguacate","mango","papaya","pina"]
datos_entrenamiento = []
datos_prueba = []
for etiqueta, fruta in enumerate(etiquetas):
    carpeta_fruta = os.path.join(carpeta_entrenamiento, fruta)
    datos_entrenamiento += cargar_imagenes_y_etiquetas(carpeta_fruta, etiqueta)
for etiqueta, fruta in enumerate(etiquetas):
    carpeta_fruta = os.path.join(carpeta_prueba, fruta)
    datos_prueba += cargar_imagenes_y_etiquetas(carpeta_fruta, etiqueta)

x_train = np.array([datos[0] for datos in datos_entrenamiento])
y_train = np.array([datos[1] for datos in datos_entrenamiento])
x_test = np.array([datos[0] for datos in datos_prueba])
y_test = np.array([datos[1] for datos in datos_prueba])

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

img_rows, img_cols = 100, 100
img_channels = 3

print(x_train[0])

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)

x_train = x_train.astype('float32') / 255  
x_test = x_test.astype('float32') / 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_train[0])

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train[1000:1020])

batch_size = 128
epochs = 4
input_shape = (img_rows,img_cols,img_channels)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

p = model.predict(x_test)

def get_numbers(predictions):
  l = []
  for p in predictions:
    l.append(np.argmax(p))
  return np.array(l)

print(p[:5])
p2 = get_numbers(p)
print(p2)

def plot_samples_with_predictions(x, y_true, y_pred, n=40):
    filas = (n + 9) // 10 
    for i in range(n):
        plt.subplot(filas, 10, i+1)
        plt.imshow(x[i])
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Q: {[y_true[i]]}\nP: {[y_pred[i]]}')
        print("Q: ", etiquetas[y_true[i]], "P: ", etiquetas[y_pred[i]])
predicted_labels = np.argmax(p, axis=1)
plot_samples_with_predictions(x_test, np.argmax(y_test, axis=1), predicted_labels)
plt.show()
