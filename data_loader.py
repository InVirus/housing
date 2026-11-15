import os
import numpy as np

def generate_postwar_housing_dataset(n_samples=5000, random_state=42):
    np.random.seed(random_state)

    region_safety = np.random.uniform(0.3, 1.0, n_samples)
    infra_recovery = np.clip(0.4 * region_safety + np.random.uniform(0.2, 0.6, n_samples), 0.1, 1.0)
    shelter_distance = np.random.exponential(scale=200, size=n_samples) + 30
    shelter_distance = np.clip(shelter_distance, 30, 800)
    energy_autonomy = np.random.choice([0.1, 0.4, 0.8, 1.0], size=n_samples, p=[0.25, 0.35, 0.25, 0.15])
    area_sqm = np.random.normal(loc=65, scale=20, size=n_samples)
    area_sqm = np.clip(area_sqm, 28, 140)
    rooms = np.clip(np.round(area_sqm / 28), 1, 4)
    new_building = np.random.binomial(1, p=0.45, size=n_samples)
    economic_activity = np.clip(0.5 * region_safety + np.random.uniform(0.1, 0.5, n_samples), 0.2, 1.0)
    recovery_quarter = np.random.randint(1, 17, n_samples)

    base_price_sqm = 750.0
    price_sqm = (
        base_price_sqm +
        580.0 * region_safety +
        340.0 * infra_recovery +
        280.0 * energy_autonomy +
        220.0 * new_building +
        310.0 * economic_activity +
        22.0 * recovery_quarter -
        0.18 * shelter_distance +
        np.random.normal(0, 45, n_samples)
    )
    price_sqm = np.clip(price_sqm, 600.0, 3200.0)

    total_price_kusd = np.round((price_sqm * area_sqm) / 1000.0, 2)
    price_sqm = np.round(price_sqm, 1)
    area_sqm = np.round(area_sqm, 1)
    shelter_distance = np.round(shelter_distance, 0)
    region_safety = np.round(region_safety, 2)
    infra_recovery = np.round(infra_recovery, 2)
    economic_activity = np.round(economic_activity, 2)

    header = [
        "RegionSafety",
        "InfraRecovery",
        "ShelterDistance",
        "EnergyAutonomy",
        "AreaSqm",
        "Rooms",
        "NewBuilding",
        "EconomicActivity",
        "RecoveryQuarter",
        "TotalPriceKUSD"
    ]

    data = np.column_stack([
        region_safety,
        infra_recovery,
        shelter_distance,
        energy_autonomy,
        area_sqm,
        rooms,
        new_building,
        economic_activity,
        recovery_quarter,
        total_price_kusd
    ])

    n_train = int(0.8 * n_samples)
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    train_data = data[train_indices]
    test_data = data[test_indices]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "train.csv")
    test_path = os.path.join(base_dir, "test.csv")

    fmt = ["%.2f", "%.2f", "%.0f", "%.2f", "%.1f", "%.0f", "%.0f", "%.2f", "%.0f", "%.2f"]
    header_str = ",".join(header)

    np.savetxt(train_path, train_data, delimiter=",", header=header_str, comments="", fmt=fmt)
    np.savetxt(test_path, test_data, delimiter=",", header=header_str, comments="", fmt=fmt)

    print(f"Згенеровано набір даних:")
    print(f" - Навчальна вибірка: {len(train_data)} записів -> {train_path}")
    print(f" - Тестова вибірка: {len(test_data)} записів -> {test_path}")
    return train_path, test_path

if __name__ == "__main__":
    generate_postwar_housing_dataset()
