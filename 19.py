import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers, utils
from sklearn.metrics import log_loss


# Данные

X = np.array([
    [1, 2, 3, 4, 5],
    [0, 1, 0, 1, 0],
    [2, 1, 2, 1, 2],
    [5, 4, 3, 2, 1]
])
y = np.array([0, 1, 0, 2])

# One-hot кодировка целевых меток
y_cat = utils.to_categorical(y, num_classes=3)



# Модель

def make_model():
    """
    Создаёт простую нейросеть:
    - скрытый слой: 10 нейронов, ReLU
    - выходной слой: 3 нейрона, softmax
    """
    model = models.Sequential([
        layers.Dense(10, activation="relu", input_shape=(5,)),
        layers.Dense(3, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    return model


# Обучение с шумом

def train_with_noise(std: float) -> float:
    """
    Обучает модель на данных с добавленным гауссовым шумом.
    Параметры:
        std (float): стандартное отклонение шума
    Возвращает:
        log_loss (float): значение функции потерь на исходных данных
    """
    model = make_model()
    X_noisy = X + np.random.normal(0, std, X.shape)
    model.fit(X_noisy, y_cat, epochs=50, verbose=0)
    preds = model.predict(X, verbose=0)
    return log_loss(y, preds)


# Тестовый варинт ради эксперимента

std_values = [0.01, 0.03, 0.3]
losses = [train_with_noise(std) for std in std_values]

# Вывод результатов в консоль
print("Результаты эксперимента:")
for std, loss in zip(std_values, losses):
    print(f"std={std:.2f} → log_loss={loss:.4f}")



# Визуализация нашего кода

fig, ax = plt.subplots(figsize=(6, 3))

for i, loss in enumerate(losses):
    rect = plt.Rectangle((i, 0), 1, 1, color=plt.cm.coolwarm(loss / max(losses)))
    ax.add_patch(rect)
    ax.text(i + 0.5, 0.5, f"{loss:.4f}", ha="center", va="center", color="black")

ax.set_xlim(0, len(std_values))
ax.set_ylim(0, 1)
ax.set_xticks([i + 0.5 for i in range(len(std_values))])
ax.set_xticklabels([f"std={s}" for s in std_values])
ax.set_yticks([])
ax.set_title("Log loss при разных std")
ax.set_xlabel("Уровень шума")
ax.set_ylabel("Условная шкала")

plt.show()
