import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
# from tensorflow.keras.utils import plot_model

# 定义数据增强。增强模型泛化能力，模拟真实场景，提升模型性能
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(    # 预处理
    rescale=1. / 255,           # 对像素值进行缩放
    horizontal_flip=True,       # 随机水平反转
    rotation_range=45           # 随机角度角度
)
# 导入训练集，生成训练数据批次
train_generator = train_datagen.flow_from_directory(    # 预处理
    'data_1_train',
    target_size=(48, 48),   # 设置图片大小
    batch_size=32,          # 每批次包含32张图片
    class_mode='sparse',    # 稀疏标签类型，即标签为整数，提高效率
    color_mode='grayscale'  # 灰度
)

# 验证集
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
# 导入验证集，生成验证数据批次
val_generator = val_datagen.flow_from_directory(
    'data_2_val',
    target_size=(48, 48),
    batch_size=32,
    class_mode='sparse',
    color_mode='grayscale'
)

# 定义模型架构。构建一个CNN模型，用于图像分类。
""" 输入形状为 (48, 48, 1) 的灰度图像。
    通过三个卷积层（Conv2D）提取特征，每个卷积层后接最大池化层（MaxPooling2D）以减少维度。
    将三维特征图展平（Flatten）为一维向量。
    添加一个全连接层（Dense），并使用 Dropout 防止过拟合。
    最后一层输出 5 个类别的概率分布（softmax）。"""
# Sequential 是 Keras 中用于构建神经网络模型的一种线性堆叠模型，层与层之间按顺序连接，每一层的输出直接作为下一层的输入
model = models.Sequential()
# 第一个卷积层，包含 32 个过滤器，每个过滤器大小为 (3, 3)。激活函数为 ReLU，用于引入非线性，可以加速收敛并环节梯度消失的问题。
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
# 第一个大池化层，池化窗口大小为 (2, 2)。通过取局部区域的最大值来降低特征图的空间维度。
model.add(layers.MaxPooling2D((2, 2)))
# 第二个卷积层，包含 64 个过滤器，每个过滤器大小为 (3, 3)
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# 第二个大池化层，池化窗口大小为 (2, 2)，进一步降低特征图的空间维度。
model.add(layers.MaxPooling2D((2, 2)))
# 第三个卷积层，包含 128 个过滤器，每个过滤器大小为 (3, 3)，继续提取更高层次的特征，增强模型表达能力
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# 第三个大池化层，池化窗口大小为 (2, 2)，进一步降低特征图的空间维度。
model.add(layers.MaxPooling2D((2, 2)))
# 展平层，将三维特征图展平为一维向量，以便输入到全连接层。
model.add(layers.Flatten())
# 全连接层，包含 128 个神经元，激活函数为 ReLU，用于进一步增强模型表达能力。
model.add(layers.Dense(128, activation='relu'))
# Dropout层，防止过拟合，随机丢弃50%的神经元节点，以减少模型敏感度。
model.add(layers.Dropout(0.5))
# 输出层，包含 5 个神经元，对应 5 类分类任务，激活函数为 softmax，用于将输出转换为概率分布
model.add(layers.Dense(5, activation='softmax'))

# 绘制模型结构图并保存为文件
# plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)
# 绘图失败，原因不明

# 控制台绘制模型结构
model.summary()

# 编译模型,配置训练参数
""" 优化器使用Adam算法，自动调整学习率以加速收敛。
    损失函数选择稀疏分类交叉熵，适用于多分类问题，标签为整数形式。
    评估指标设置为准确率，用于衡量模型预测的正确性。"""
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
""" model.fit：启动模型训练。
    train_generator：训练数据生成器，提供训练数据。
    steps_per_epoch：每个epoch的训练步数，等于训练样本总数除以批量大小。
    epochs=50：总共训练50个epoch。
    validation_data：验证数据生成器，提供验证数据。
    validation_steps：每个epoch的验证步数，等于验证样本总数除以批量大小。
    返回值history记录了训练过程中的损失和指标。"""
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=100,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size
)

# 评估模型
test_loss, test_acc = model.evaluate(val_generator)
print(f'Test accuracy: {test_acc}')
print(f'Test loss: {test_loss}')

# 保存模型
model.save(r'models/emotion_recognition_model_v1.8.keras')

# 绘制训练过程中的准确率和损失曲线
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

