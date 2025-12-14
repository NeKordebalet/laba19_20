from dvadcatoe import build_auto_model, build_manual_model, train_and_evaluate

import numpy as np

if __name__ == "__main__":
    # Автоматическая инициализация
    auto_model = build_auto_model()
    acc_auto = train_and_evaluate(auto_model, "Автоматическая инициализация")

    # Ручная инициализация (ортогональная матрица)
    custom_matrix = np.linalg.qr(np.random.randn(784, 128))[0].astype("float32")
    manual_model = build_manual_model(custom_matrix)
    acc_manual = train_and_evaluate(manual_model, "Ручная инициализация")

    # Итоговое сравнение
    print("\nСравнение моделей:")
    print(f"Автоматическая инициализация → точность: {acc_auto:.4f}")
    print(f"Ручная инициализация         → точность: {acc_manual:.4f}")
