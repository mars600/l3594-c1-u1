import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# Load the provided files from the given URLs
ground_truth_url = "https://raw.githubusercontent.com/mars600/l3594-c1-u1/refs/heads/main/Data/network_traffic_data.csv"
ecod_url = "https://raw.githubusercontent.com/mars600/l3594-c1-u1/refs/heads/main/detect_ecod.txt"
iforest_url = "https://raw.githubusercontent.com/mars600/l3594-c1-u1/refs/heads/main/detect_iforest.txt"

# Read the data
ground_truth = pd.read_csv(ground_truth_url)
ecod_results = pd.read_csv(ecod_url, delimiter="\t")
iforest_results = pd.read_csv(iforest_url, delimiter="\t")

# Display the first few rows of each dataset to check if they loaded correctly
print("Ground truth:")
print(ground_truth.head())
print("ECOD results:")
print(ecod_results.head())
print("IForest results:")
print(iforest_results.head())

# Assuming the ground truth has columns like "timestamp", "src_ip", "dest_ip" and "true_label"
# We can join the results from ECOD and IForest with the ground truth using shared identifiers
# Adjust column names based on the actual structure of the files

# Join ECOD results with ground truth
ecod_combined = pd.merge(ground_truth, ecod_results, on=["timestamp", "src_ip", "dest_ip"], how="left")
iforest_combined = pd.merge(ground_truth, iforest_results, on=["timestamp", "src_ip", "dest_ip"], how="left")

# Assuming the ground truth file has a 'true_label' column with 1 for anomaly and 0 for normal
# For ECOD, let's compute precision, recall, and F1 score
ecod_precision, ecod_recall, ecod_f1, _ = precision_recall_fscore_support(
    ecod_combined["true_label"], ecod_combined["predicted_label"], average="binary"
)

# For IForest, compute precision, recall, and F1 score
iforest_precision, iforest_recall, iforest_f1, _ = precision_recall_fscore_support(
    iforest_combined["true_label"], iforest_combined["predicted_label"], average="binary"
)

# Display the results
print(f"ECOD Precision: {ecod_precision}, Recall: {ecod_recall}, F1: {ecod_f1}")
print(f"IForest Precision: {iforest_precision}, Recall: {iforest_recall}, F1: {iforest_f1}")
