import matplotlib.pyplot as plt
import matplotlib.patches as patches

'''
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='softmax'))
'''


def model_structure1():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    layers = [
        ("Input Layer\n(48x48x1)", (2.5, 9), "#d6eaff"),
        ("Conv2D(32)\nReLU", (2.5, 8), "#ffe0b3"),
        ("MaxPooling2D", (2.5, 7), "#ffd6d6"),
        ("Conv2D(64)\nReLU", (2.5, 6), "#ffe0b3"),
        ("MaxPooling2D", (2.5, 5), "#ffd6d6"),
        ("Conv2D(128)\nReLU", (2.5, 4), "#ffe0b3"),
        ("MaxPooling2D", (2.5, 3), "#ffd6d6"),
        ("Flatten", (2.5, 2), "#e6ffe6"),
        ("Dense(128)\nReLU", (2.5, 1), "#f8d2d2"),
        ("Dropout(0.5)", (2.5, 0.5), "#f8d2d2"),
        ("Output(Dense(5))\nSoftmax", (2.5, 0), "#f2caca")
    ]

    for text, (x_center, y), color in layers:
        width = 5
        height = 0.6
        rect = patches.Rectangle((x_center - width / 2, y - height / 2), width, height,
                                 linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        plt.text(x_center, y, text, ha='center', va='center', fontsize=10)

    for i in range(len(layers) - 1):
        x_start = 2.5
        y_start = layers[i][1][1] - 0.3
        y_end = layers[i + 1][1][1] + 0.3
        plt.arrow(x_start, y_start, 0, y_end - y_start, head_width=0.1, length_includes_head=True, color='black')

    plt.title("CNN Model Structure Visualization", fontsize=14)
    plt.tight_layout()
    plt.savefig("cnn_model_structure.png", dpi=200, bbox_inches='tight')
    plt.show()


def model_structure2():
    plt.figure(figsize=(10, 6))

    layers = [
        {"name": "Input", "shape": (48, 48, 1), "type": "input"},
        {"name": "Conv2D (32,3x3)", "shape": (46, 46, 32), "type": "conv"},
        {"name": "MaxPool (2x2)", "shape": (23, 23, 32), "type": "pool"},
        {"name": "Conv2D (64,3x3)", "shape": (21, 21, 64), "type": "conv"},
        {"name": "MaxPool (2x2)", "shape": (10, 10, 64), "type": "pool"},
        {"name": "Conv2D (128,3x3)", "shape": (8, 8, 128), "type": "conv"},
        {"name": "MaxPool (2x2)", "shape": (4, 4, 128), "type": "pool"},
        {"name": "Flatten", "shape": (2048,), "type": "flatten"},
        {"name": "Dense (128)", "shape": (128,), "type": "dense"},
        {"name": "Output (5)", "shape": (5,), "type": "output"}
    ]

    for i, layer in enumerate(layers):
        color = {
            "input": "lightgreen",
            "conv": "lightblue",
            "pool": "lightcoral",
            "flatten": "violet",
            "dense": "gold",
            "output": "orange"
        }[layer["type"]]

        plt.barh(i, width=10, height=0.5, color=color, edgecolor='k')
        plt.text(5, i, f"{layer['name']}\n{layer['shape']}",
                 ha='center', va='center', fontsize=9)

    for i in range(len(layers) - 1):
        plt.arrow(10, i, 0.9, 1 - i, head_width=0.3, head_length=0.2, fc='k')

    plt.yticks([])
    plt.title("CNN Model Architecture", pad=20)
    plt.xlim(0, 12)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def model_structure3():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 层信息：("层名称", y位置, 颜色)
    layers_info = [
        ("Input Layer\n(48x48x1)", 10, "#d6eaff"),
        ("Conv2D(32)\nReLU", 9, "#ffe0b3"),
        ("MaxPooling2D", 8, "#ffd6d6"),
        ("Conv2D(64)\nReLU", 7, "#ffe0b3"),
        ("MaxPooling2D", 6, "#ffd6d6"),
        ("Conv2D(128)\nReLU", 5, "#ffe0b3"),
        ("MaxPooling2D", 4, "#ffd6d6"),
        ("Flatten", 3, "#e6ffe6"),
        ("Dense(128)\nReLU", 2, "#f8d2d2"),
        ("Dropout(0.5)", 1.5, "#f8d2d2"),
        ("Output(Dense(5))\nSoftmax", 1, "#f2caca")
    ]

    x_center = 5  # 居中绘制
    rect_width = 5
    rect_height = 0.6

    # 绘制每一层
    for text, y, color in layers_info:
        rect = patches.Rectangle((x_center - rect_width / 2, y - rect_height / 2),
                                 rect_width, rect_height,
                                 linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        plt.text(x_center, y, text, ha='center', va='center', fontsize=10)

    # 绘制箭头（从上到下）
    for i in range(len(layers_info) - 1):
        y_start = layers_info[i][1] - rect_height / 2
        y_end = layers_info[i + 1][1] + rect_height / 2
        plt.arrow(x_center, y_start, 0, y_end - y_start,
                  head_width=0.1, length_includes_head=True, color='black')

    plt.title("CNN Model Structure Visualization (Top to Bottom)", fontsize=14)
    plt.tight_layout()
    plt.savefig("cnn_model_structure_top_to_bottom.png", dpi=200, bbox_inches='tight')
    plt.show()


model_structure1()
