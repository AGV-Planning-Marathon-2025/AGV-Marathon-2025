import pandas as pd

# Load DataFrame from pickle
df = pd.read_pickle("../car_ros2/car_ros2/recorded_races/racedata_mpc_only_neg.pkl")

# Save to CSV
df.to_csv("data/output.csv", index=False)
print("Saved as output.csv")