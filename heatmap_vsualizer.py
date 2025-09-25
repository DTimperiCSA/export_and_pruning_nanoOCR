import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

tm_matrix = np.load(r"C:\Users\andre\Desktop\export_and_pruning_nanoOCR\results\tm_similarity_matrix.npy")

plt.figure(figsize=(10,8))
sns.heatmap(tm_matrix, cmap="viridis")
plt.title("TM Similarity Heatmap")
plt.show()
