import pandas as pd
import numpy as np

# Function to generate synthetic data
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    
    data = {
        "device_id": 1233,
        "temperature": np.random.normal(21, 0, num_samples),
        "pressure": np.random.normal(100, 0, num_samples),
        "voltage": np.random.normal(3, 0, num_samples),
        "CO2": np.random.normal(400, 0, num_samples),
        "rotation_speed": np.random.normal(200, 0, num_samples),
        "label": np.random.choice([0, 1], num_samples, p=[0.95, 0.05])  # 5% anomalies
    }

    # Introduce some anomalies
    for col in ["temperature", "pressure", "voltage", "CO2", "rotation_speed"]:
        data[col][data["label"] == 1] *= np.random.uniform(1.5, 2.0, sum(data["label"] == 1))

    return pd.DataFrame(data)

# Generate and save the dataset
df = generate_synthetic_data()
df.to_csv('anomaly_detection_dataset.csv', index=False)
print(df.head())
