import tensorflow as tf
import cv2
import numpy as np
from deepface import DeepFace


def emo_model(img_path):
    result = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=False)
    main_emo = result[0]['dominant_emotion']
    conf = result[0]['face_confidence']
    return main_emo, conf


def emo_model2():
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
    # print("predictions:", predictions)      # [[0.18924606 0.10887002 0.88397245 0.11359134 0.00432014]]
    predicted_class = np.argmax(predictions, axis=1)[0]
    # print("predicted_class:", predicted_class)       # 2,即第三个 （happy,sad,angry,surprise,nature）
    emo_class = emotion_dict[predicted_class]
    predicted_emotion = emotion_dict[predicted_class]
    # print("predicted_emotion:", predicted_emotion)      # angry
    confidence = predictions[0]
    emo_conf = max(confidence)
    emo_conf = float(emo_conf)
    emo_conf_percent = round(emo_conf * 100, 2)

    # emo_conf_round =
    # print(f'Predicted emotion: {predicted_emotion}')
    # print('Confidence for each class:')
    # for i, conf in enumerate(confidence):
    #     print(f'{emotion_dict[i]}: {conf:.4f}')

    print(f"有{emo_conf_percent:.2f}%的概率，人物表情是{emo_class}")

    return emo_conf, emo_class


if __name__ == '__main__':
    emo_model2()
