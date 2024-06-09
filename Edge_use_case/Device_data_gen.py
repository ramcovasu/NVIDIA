import random
from datetime import datetime, timedelta
import time
import csv


def generate_synthetic_data():
    """Generates a single row of synthetic device data with anomaly control.

    Returns:
        A dictionary containing the generated data:
            device_id (str): The constant device ID (hardcoded to 1233).
            datetime (str): Date and time in YYYY-MM-DD HH:MM:SS format.
            temperature (int): Random temperature value.
            pressure (int): Random pressure value.
            CO2(int): Random value (3 to 5).
            rotation_speed (int): Random rotation speed.
            For temperature normal range is 20 to 25, for pressure 100 to 105, for CO2 400 to 500, voltage 3 to 4
    """

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    temperature = random.uniform(20, 25)  # Normal range
    pressure = random.uniform(100, 105)  # Normal range
    CO2 = random.uniform(400, 500)  # Normal range

    # Introduce anomaly with 5% chance every 20 rows (adjustable)
    if random.random() < 0.05 and row_count % 20 == 0:
        temperature = random.uniform(41, 50)
        pressure = random.uniform(150, 200)
        CO2 = random.uniform(500, 600)

    return {
        "device_id": "1233",  # Hardcoded device ID
        "datetime": current_time,
        "temperature": round(temperature),
        "pressure": round(pressure),
        # Add rotation_speed if needed (adjust range),
        "CO2":round(CO2),
        "voltage":round(random.uniform(3,4)),
        "rotation_speed": round(random.uniform(100, 200))
    }


if __name__ == "__main__":
    row_count = 0
    output_file = "device_data.csv"  # Hardcoded output file path

    # Open the CSV file in append mode
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "device_id", "datetime", "temperature", "pressure","CO2","voltage", "rotation_speed"])

        # Write header row only if file is empty
        csvfile.seek(0, 2)
        if csvfile.tell() == 0:
            writer.writeheader()

        while True:
            start_time = time.time()  # Track loop start time

        # Generate and write 20 records
            for _ in range(20):
                data = generate_synthetic_data()
                writer.writerow(data)
                row_count += 1

            # Calculate remaining time in the minute
            elapsed_time = time.time() - start_time
            sleep_time = max(0, 60 - elapsed_time)  # Ensure at least 60 seconds

            # Sleep for remaining time
            time.sleep(sleep_time)

            # Add user interrupt handling (e.g., keyboard interrupt)
            if input("Press Enter to continue, any other key to stop: ") != "":
                break