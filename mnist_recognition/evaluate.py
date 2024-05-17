from data import load_and_preprocess_data
from tensorflow.keras.models import load_model

def main():
    # Charger et prétraiter les données
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    # Charger le modèle
    model = load_model('mnist_model.h5')

    # Évaluer le modèle
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_acc}")

if __name__ == "__main__":
    main()
