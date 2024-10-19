from scipy.stats import ks_2samp

# Function to detect drift
def detect_drift(new_data, reference_data, threshold=0.05):
    drift_detected = False
    for col in reference_data.columns:
        stat, p_value = ks_2samp(reference_data[col], new_data[col])
        if p_value < threshold:  # Drift is detected if p-value < threshold
            drift_detected = True
            print(f"Drift detected in feature: {col}")
            break
    return drift_detected

