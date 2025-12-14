import tensorflow as tf
import numpy as np
import time


# Данные

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Преобразование и нормализация
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0



# Модель с автоинициализацией

def build_auto_model():
    """
    Создаёт модель с автоматической инициализацией весов:
    - скрытый слой: 128 нейронов, ReLU
    - выходной слой: 10 классов, softmax
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu", input_shape=(784,)),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model



# Кастомный слой с ручной инициализацией

class ManualInitLayer(tf.keras.layers.Layer):
    """
    Слой Dense с ручной инициализацией весов.
    """
    def __init__(self, units, custom_weights, activation=None):
        super().__init__()
        self.units = units
        self.custom_weights = custom_weights
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.w = tf.Variable(self.custom_weights, trainable=True, dtype=tf.float32)
        self.b = tf.Variable(tf.zeros([self.units]), trainable=True)

    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w) + self.b)


def build_manual_model(custom_matrix):
    """
    Создаёт модель с ручной инициализацией весов.
    """
    model = tf.keras.Sequential([
        ManualInitLayer(128, custom_matrix, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model



# Функция обучения и оценки

def train_and_evaluate(model, name):
    """
    Обучает модель и выводит время обучения и точность на тесте.
    """
    start = time.time()
    model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=0)
    end = time.time()
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{name}: время={round(end - start, 2)} сек, точность={acc:.4f}")
    return 
     

    # Точка входа (main)

if __name__ == "__main__":
    main()