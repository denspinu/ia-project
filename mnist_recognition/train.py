# train.py
from data import load_and_preprocess_data, display_sample_images
from model import create_model

def main():
    # Charger et prétraiter les données
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    # Afficher quelques exemples de chiffres
    display_sample_images(x_train, y_train)

    # Construire le modèle
    model = create_model()

    # Entraîner le modèle
    model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

    # Sauvegarder le modèle
    model.save('mnist_model.h5')

if __name__ == "__main__":
    main()