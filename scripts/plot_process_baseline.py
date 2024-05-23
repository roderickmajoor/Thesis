import matplotlib.pyplot as plt

# Data from the tables (excluding Total Edit Distance)
baseline_data = {
    'WER': 0.4243,
    'CER': 0.3422,
    'Recall': 0.9475,
    'Precision': 0.7403
}

post_process_data = {
    'WER': 0.3916,
    'CER': 0.3169,
    'Recall': 0.9418,
    'Precision': 0.7622
}

metrics = list(baseline_data.keys())
baseline_values = list(baseline_data.values())
post_process_values = list(post_process_data.values())

# Plotting
x = range(len(metrics))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x, baseline_values, width, label='Baseline')
rects2 = ax.bar([i + width for i in x], post_process_values, width, label='Post Process')

ax.set_ylabel('Scores')
ax.set_title('Comparison of HTR system with and without post-processing')
ax.set_xticks([i + width/2 for i in x])
ax.set_xticklabels(metrics)
ax.legend()

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
