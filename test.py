"""这里跑一下测试，这里代码不执行"""
import tensorflow as tf
import numpy as np
import cv2


def main():
    model = tf.keras.models.load_model(r'models\emotion_recognition_model_v1.5.keras')

    emotion_dict = {
        0: 'happy',
        1: 'sad',
        2: 'angry',
        3: 'surprise',
        4: 'nature'
    }

    image_path = input("输入图片路径：")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (48, 48))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    predictions = model.predict(image)
    print("predictions:", predictions)      # [[0.18924606 0.10887002 0.58397245 0.11359134 0.00432014]]
    predicted_class = np.argmax(predictions, axis=1)[0]
    print("predicted_class:", predicted_class)       # 2,即第三个 （happy,sad,angry,surprise,nature）
    predicted_emotion = emotion_dict[predicted_class]
    print("predicted_emotion:", predicted_emotion)      # angry
    confidence = predictions[0]

    print(f'Predicted emotion: {predicted_emotion}')
    print('Confidence for each class:')
    for i, conf in enumerate(confidence):
        print(f'{emotion_dict[i]}: {conf:.4f}')


if __name__ == '__main__':
    main()
