# data.py
import tensorflow as tf

def load_and_preprocess_data():
    from tensorflow.keras.datasets import mnist

    # Charger les données
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normaliser les images de 0-255 à 0-1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Redimensionner les images pour qu'elles soient 28x28x1
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # Convertir les labels en encodage one-hot
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

def display_sample_images(x_train, y_train):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(x_train[i].squeeze(), cmap='gray')
        plt.title(f"Label: {y_train[i].argmax()}")
        plt.axis('off')
    plt.show()