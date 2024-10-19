import pandas as pd
import numpy as np

def generate_data(num_samples=100, drift=False):
    np.random.seed(42)
    data = {
        'sepal_length': np.random.normal(5.0, 0.5, num_samples),
        'sepal_width': np.random.normal(3.5, 0.3, num_samples),
        'petal_length': np.random.normal(1.4, 0.5, num_samples),
        'petal_width': np.random.normal(0.2, 0.1, num_samples)
    }
    if drift:
        data['sepal_length'] = np.random.normal(7.0, 0.5, num_samples)
        data['petal_length'] = np.random.normal(2.0, 0.5, num_samples)
    return pd.DataFrame(data)

# Save the datasets if needed
reference_data = generate_data(num_samples=100, drift=False)
reference_data.to_csv('reference_data.csv', index=False)
