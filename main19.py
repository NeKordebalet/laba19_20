from exper import train_with_noise, visualize_results

if __name__ == "__main__":
    # Эксперимент с разными значениями std
    std_values = [0.01, 0.03, 0.3]
    losses = [train_with_noise(std) for std in std_values]

    # Вывод результатов в консоль
    print("Результаты эксперимента:")
    for std, loss in zip(std_values, losses):
        print(f"std={std:.2f} → log_loss={loss:.4f}")

    # Визуализация
    visualize_results(std_values, losses)
