import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os # To ensure the output directory exists

# --- 1. Define the Confusion Matrix Data ---
# Use the data from your second, successful run
confusion_matrix_data = np.array([
    [469384235,  34412911,   8975163,  14968580,  17364729,  24548990,  45596074],
    [ 17990013, 101440263,    617418,    104331,    229728,    872308,   1551730],
    [ 19649909,    910397,  54522947,    152556,   1619998,   1493427,   1241266],
    [ 19463838,    482093,    226865, 164656789,   3304733,   3237195,   8196303],
    [  9146946,   1301221,    560790,   2355463,  54144998,    589864,   6283851],
    [ 29877222,    179233,    515069,   1749912,   2473108,  87384319,   3436784],
    [ 67895784,   2495562,   2130912,   6526394,  15401836,  22067000, 370565214]
], dtype=np.int64) # Ensure integer type if needed, though it will be normalized

# --- 2. Define Class Names (Ensure order matches the matrix rows/columns) ---
class_names = ['background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural']

# --- 3. Normalize the Confusion Matrix (Row-wise percentages) ---
# Calculate the sum of each row (total true instances for each class)
row_sums = confusion_matrix_data.sum(axis=1)[:, np.newaxis]

# Avoid division by zero for classes that might not appear in GT (though unlikely here)
# Divide each element by its row sum to get percentages. Use np.nan_to_num to handle potential 0/0 -> NaN.
norm_cm = np.nan_to_num(confusion_matrix_data.astype(float) / row_sums)

# --- 4. Create the Heatmap Plot ---
plt.figure(figsize=(12, 10)) # Adjust figure size as needed

# Use seaborn heatmap function
sns.heatmap(
    norm_cm,
    annot=True,           # Show the percentage values on the heatmap cells
    fmt=".2%",            # Format annotations as percentages with 2 decimal places
    cmap='Blues',         # Colormap (e.g., 'Blues', 'YlGnBu', 'viridis')
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=.5,        # Add lines between cells
    cbar=True             # Show the color bar
)

# --- 5. Add Labels and Title ---
plt.xlabel('Predicted Labels', fontsize=12)
plt.ylabel('True Labels', fontsize=12)
plt.title('Normalized Confusion Matrix (LoveDA Dataset - UNetMambaCA)', fontsize=14)

# Rotate labels for better readability if needed
plt.xticks(rotation=45, ha='right') # Rotate x-axis labels
plt.yticks(rotation=0)             # Keep y-axis labels horizontal

# --- 6. Adjust Layout and Save/Show ---
plt.tight_layout() # Adjust plot to prevent labels overlapping

# Define output directory and ensure it exists
output_dir = './output_loveda_hunxiao' # Same directory as your .npy/.csv files
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, 'confusion_matrix_heatmap_normalized.png')

plt.savefig(save_path, dpi=300) # Save the figure with high resolution
print(f"混淆矩阵热力图已保存至: {save_path}")

plt.show() # Display the plot

# --- Optional: Plot Raw Counts (Uncomment if you prefer raw numbers) ---
# Note: Raw counts might be hard to read with these large numbers.
# plt.figure(figsize=(12, 10))
# sns.heatmap(
#     confusion_matrix_data,
#     annot=True,
#     # Choose a format: 'd' for integer, '.1e' for scientific notation
#     fmt='d',
#     cmap='Blues',
#     xticklabels=class_names,
#     yticklabels=class_names,
#     linewidths=.5,
#     cbar=True
# )
# plt.xlabel('Predicted Labels', fontsize=12)
# plt.ylabel('True Labels', fontsize=12)
# plt.title('Confusion Matrix (Raw Counts - LoveDA Dataset - UNetMambaCA)', fontsize=14)
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=0)
# plt.tight_layout()
# save_path_raw = os.path.join(output_dir, 'confusion_matrix_heatmap_raw.png')
# plt.savefig(save_path_raw, dpi=300)
# print(f"原始计数混淆矩阵热力图已保存至: {save_path_raw}")
# plt.show()