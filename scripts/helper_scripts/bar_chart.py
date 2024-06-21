import pandas as pd
import matplotlib.pyplot as plt

# Data for Table 1: IoU Threshold 0.5
data_05 = {
    'Mean IoU': 0.627,
    'Mean Precision': 0.416,
    'Mean Recall': 0.459,
    'Mean F1 score': 0.434
}

# Data for Table 2: IoU Threshold 0.25
data_025 = {
    'Mean IoU': 0.473,
    'Mean Precision': 0.804,
    'Mean Recall': 0.898,
    'Mean F1 score': 0.844
}

# Data for Table 3: IoU Threshold >0
data_gt0 = {
    'Mean IoU': 0.182,
    'Mean Precision': 0.845,
    'Mean Recall': 0.946,
    'Mean F1 score': 0.887
}

# Convert data to DataFrame
df = pd.DataFrame([data_05, data_025, data_gt0], index=['IoU 0.5', 'IoU 0.25', 'IoU >0'])

# Plotting
df.plot(kind='bar', figsize=(10, 6))
plt.title('Comparison of Mean IoU, Precision, Recall, and F1 Scores at Different IoU Thresholds')
plt.xlabel('IoU Threshold')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.legend(title='Metrics')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
