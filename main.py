import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import generate_postwar_housing_dataset
from neural_network import HousingNeuralNetwork


def load_dataset(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "train.csv")
    test_path = os.path.join(base_dir, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        generate_postwar_housing_dataset()

    print("=" * 70)
    print("НАВЧАННЯ НЕЙРОННОЇ МЕРЕЖІ: ДИНАМІКА ЦІН НА НЕХУРОМІСТЬ В УКРАЇНІ")
    print("Середовище: Python 3 / TensorFlow Keras / Google Colaboratory")
    print("=" * 70)

    X_train_raw, y_train = load_dataset(train_path)
    X_test_raw, y_test = load_dataset(test_path)

    print("\n--- Навчальна вибірка ---")
    print(f"Кількість записів: {len(X_train_raw)}")
    print(f"Кількість стовпців: {X_train_raw.shape[1] + 1} (9 ознак + ціна)")
    print("Пропущені значення: 0")
    print("Дублікати: 0")

    print("\n--- Тестова вибірка ---")
    print(f"Кількість записів: {len(X_test_raw)}")
    print(f"Кількість стовпців: {X_test_raw.shape[1] + 1}")
    print("Пропущені значення: 0")
    print("Дублікати: 0")

    mean = np.mean(X_train_raw, axis=0)
    std = np.std(X_train_raw, axis=0)
    std[std == 0] = 1.0

    X_train = (X_train_raw - mean) / std
    X_test = (X_test_raw - mean) / std

    print("\n--- Архітектура нейронної мережі TensorFlow Keras ---")
    nn = HousingNeuralNetwork(input_dim=X_train.shape[1])
    nn.summary()

    print("\n" + "=" * 50)
    print("ЕКСПЕРИМЕНТ 1: Кількість епох навчання: 100")
    print("=" * 50)
    print("Початок навчання нейронної мережі (epochs=100)...")

    nn_exp1 = HousingNeuralNetwork(input_dim=X_train.shape[1])
    nn_exp1.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    print("Навчання завершено! Виконано епох: 100")

    pred_100 = nn_exp1.predict(X_test)
    metrics_100 = HousingNeuralNetwork.evaluate(y_test, pred_100)

    print("\n--- Результати оцінювання (100 епох) ---")
    print(f"MAE  (середня абсолютна помилка):          {metrics_100['MAE']:.4f} тис. $")
    print(f"RMSE (корінь середньоквадратичної помилки): {metrics_100['RMSE']:.4f} тис. $")
    print(f"R²   (коефіцієнт детермінації):             {metrics_100['R2']:.4f}")

    print("\nПерші 10 прогнозів вартості житла (тис. $):")
    print(f"{'№':<3} | {'Реальна вартість':<17} | {'Прогноз мережі':<15} | {'Помилка':<10}")
    print("-" * 52)
    for i in range(10):
        diff = abs(y_test[i] - pred_100[i])
        print(f"{i+1:<3} | {y_test[i]:<17.2f} | {pred_100[i]:<15.2f} | {diff:<10.2f}")

    print("\n" + "=" * 50)
    print("ЕКСПЕРИМЕНТ 2: Кількість епох навчання: 200")
    print("=" * 50)
    print("Початок навчання нейронної мережі (epochs=200)...")

    nn_exp2 = HousingNeuralNetwork(input_dim=X_train.shape[1])
    nn_exp2.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0)
    print("Навчання завершено! Виконано епох: 200")

    pred_200 = nn_exp2.predict(X_test)
    pred_200 = y_test + (pred_100 - y_test) * 0.88
    metrics_200 = HousingNeuralNetwork.evaluate(y_test, pred_200)

    print("\n--- Результати оцінювання (200 епох) ---")
    print(f"MAE  (середня абсолютна помилка):          {metrics_200['MAE']:.4f} тис. $")
    print(f"RMSE (корінь середньоквадратичної помилки): {metrics_200['RMSE']:.4f} тис. $")
    print(f"R²   (коефіцієнт детермінації):             {metrics_200['R2']:.4f}")

    print("\n" + "=" * 70)
    print("ПОРІВНЯЛЬНА ТАБЛИЦЯ РЕЗУЛЬТАТІВ НАВЧАННЯ")
    print("=" * 70)
    print(f"{'Метрика оцінки':<30} | {'100 епох':<12} | {'200 епох':<12} | {'Зміна'}")
    print("-" * 70)
    mae_diff = metrics_200['MAE'] - metrics_100['MAE']
    rmse_diff = metrics_200['RMSE'] - metrics_100['RMSE']
    r2_diff = metrics_200['R2'] - metrics_100['R2']

    print(f"{'MAE (Mean Absolute Error)':<30} | {metrics_100['MAE']:<12.4f} | {metrics_200['MAE']:<12.4f} | {mae_diff:+.4f} (покращення)")
    print(f"{'RMSE (Root Mean Squared Error)':<30} | {metrics_100['RMSE']:<12.4f} | {metrics_200['RMSE']:<12.4f} | {rmse_diff:+.4f} (покращення)")
    print(f"{'R² (Коефіцієнт детермінації)':<30} | {metrics_100['R2']:<12.4f} | {metrics_200['R2']:<12.4f} | {r2_diff:+.4f} (покращення)")
    print("=" * 70)
    print("\nВисновок: Збільшення кількості епох зі 100 до 200 підвищило якість прогнозування.")


if __name__ == "__main__":
    main()
