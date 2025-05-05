import tensorflow as tf
from tensorflow.keras.models import load_model

def evaluate():
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    val_generator = val_datagen.flow_from_directory(
        'data_val',
        target_size=(48, 48),
        batch_size=32,
        class_mode='sparse',
        color_mode='grayscale'
    )

    model = load_model(r'models/emotion_recognition_model_v1.6.keras')

    test_loss, test_acc = model.evaluate(val_generator)
    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_acc}')

if __name__ == '__main__':
    evaluate()