import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

data = [[2, 3], [5, 7], [8, 3], [3, 5], [7, 2], [6, 8], [1, 4], [4, 6], [9, 5], [5, 4]]

# Visualise the scatter plot of the data
plt.scatter([p[0] for p in data], [p[1] for p in data], c='green', cmap="plasma", s=50, marker="o", alpha= 0.5)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Data Scatter Plot")
plt.show()


# Elbow method to find perfect k
# possible number of clusters can be in 1 to 11
ks = np.arange(1,11)


# find the inertias for different ks
intertias = []
for k in ks :
    k_menas_temp = KMeans(n_clusters=k, init="k-means++", random_state=42)
    result = k_menas_temp.fit(data)
    intertias.append(result.inertia_)

# visualise for elbow position
plt.plot(ks, intertias, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Inertias")
plt.title("Elbow position determination")
plt.show()

# From the plot it's clear that the elbow position is 5 
k_means_type_3 = KMeans(n_clusters=5, init="k-means++", random_state=42)
result = k_means_type_3.fit(data)

# Visualise the clustered plot
plt.scatter([p[0] for p in data], [p[1] for p in data], c=result.labels_, cmap="plasma", s=50, marker="o", alpha= 1)
plt.scatter(result.cluster_centers_[:, 0], result.cluster_centers_[:, 1], c="red", marker="*", s=55)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Clustered Data Plot")
plt.show()

# Predict cluster for a custom data
pred = result.predict([[10,15]])
print(pred)