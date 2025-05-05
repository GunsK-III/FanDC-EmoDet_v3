# 训练调试，这里的代码不执行
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt

# 1. 数据增强
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    rotation_range=45,
    width_shift_range=0.2,  # 随机水平平移
    height_shift_range=0.2,  # 随机垂直平移
    zoom_range=0.2,  # 随机缩放
    brightness_range=[0.5, 1.5]  # 随机亮度调整
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

# 2. 数据生成器
train_generator = train_datagen.flow_from_directory(
    'data_train',
    target_size=(48, 48),
    batch_size=32,
    class_mode='sparse',
    color_mode='grayscale'
)

val_generator = val_datagen.flow_from_directory(
    'data_val',
    target_size=(48, 48),
    batch_size=32,
    class_mode='sparse',
    color_mode='grayscale'
)

# 3. 计算类别权重
# y_train = train_generator.classes
# class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
# class_weights = dict(enumerate(class_weights))

# 4. 构建模型
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(layers.BatchNormalization())  # 批量归一化
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))  # Dropout

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())  # 批量归一化
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))  # Dropout

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())  # 批量归一化
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))  # Dropout

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))  # Dropout
model.add(layers.Dense(5, activation='softmax'))

# 5. 编译模型
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  # 调整学习率
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. 早停法
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# 7. 训练模型
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=100,  # 增加训练轮数
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[early_stopping],  # 早停法
    # class_weight=class_weights  # 类别权重
)

# 8. 评估模型
test_loss, test_acc = model.evaluate(val_generator)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')

# 9. 保存模型
model.save(r'models/emotion_recognition_model_v2.0.keras')

# 10. 绘制训练曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.show()
