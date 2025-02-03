import logging
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray


error_logger: logging.Logger = logging.getLogger("debug_logger")
error_logger.setLevel(logging.WARNING)
error_handler: logging.FileHandler = logging.FileHandler(
    filename="error_logger.log", mode="w"
)
error_formatter = logging.Formatter(
    fmt="%(name)s %(asctime)s %(message)s\nLine: %(lineno)s"
)
error_handler.setFormatter(fmt=error_formatter)
error_logger.addHandler(hdlr=error_handler)

debug_logger: logging.Logger = logging.getLogger("debug_logger")
debug_logger.setLevel(logging.DEBUG)
debug_handler: logging.FileHandler = logging.FileHandler(
    filename="debug_logger.log", mode="w"
)
debug_formatter = logging.Formatter(fmt="%(name)s %(asctime)s %(message)s\nLine: %(lineno)s")
debug_handler.setFormatter(fmt=debug_formatter)
debug_logger.addHandler(hdlr=debug_handler)


### CLASS

class KMeans:
    
    def __init__(self, data: pd.DataFrame | NDArray, k: int):
        self.data: NDArray = np.asarray(data)
        self.k = k
        self.clusters: dict[
            tuple[np.float64], list[NDArray[np.float64]]
        ] = {}

    @staticmethod
    def euclidean_distance(
        data1: NDArray[np.float64 | np.int64],
        data2: NDArray[np.float64 | np.int64]
    ) -> np.float64 :
        """
        Method calculate Euclidean distance between two points.
        """
        return np.sqrt(np.sum(a=(data1 - data2)**2, axis=1))
      
    def kmeans_plusplus(self):
        """
        Method implements K-Means++ algorithm to initialize centroids.
        Adds centorids to clusters dictionary.
        """
        # Initializing the first centroid.
        random_centorid: NDArray[np.float64] = np.array(
            object=self.data[
                np.random.randint(
                    low=0, high=len(self.data)
                )
            ],
            dtype=np.float64
        )
        centroids: list[NDArray[np.float64]] = [random_centorid]

        for _ in range(self.k - 1):
            # Calculating distances from all centroids to data points
            distances: NDArray = [
                (
                    self.euclidean_distance(
                        data1=centroids[ind],
                        data2=self.data
                    )
                ) for ind in range(len(centroids))
            ]
            
            min_distances: NDArray = np.min(a=np.asarray(distances), axis=0)
            debug_logger.debug(
                msg=f"Min distances: {min_distances}"
            )
        
            next_centroid_probabilities: NDArray[np.float64] = (
                min_distances / np.sum(min_distances)
            )

            # Sorted positions of probabilities' indexes
            sorted_probabilities_inds: NDArray[np.int64] = np.argsort(
                next_centroid_probabilities
            )
            debug_logger.debug(
                msg=f"sorted probabilities' indexes: {sorted_probabilities_inds}"
            )
            
            sorted_probabilities: NDArray[np.float64] = (
                next_centroid_probabilities[sorted_probabilities_inds]
            )
            cumulative_probabilities: NDArray[np.float64] = (
                np.cumsum(sorted_probabilities)
            )
            
            random_n: float = np.random.rand()
            # Points have the same indexes as their probabilities since
            # order has not changed.
            ind_of_point_ind: int = (
                np.searchsorted(a=cumulative_probabilities, v=random_n)
            ) # index of a point's index
            next_centroid: NDArray[np.float64] = self.data[
                sorted_probabilities_inds[ind_of_point_ind]
            ]
            
            centroids.append(next_centroid)
        
        debug_logger.debug(msg=f"List of centroids: {centroids}")

        for centroid in centroids:
            # NDArray unhashable object
            self.clusters[tuple(centroid)] = []

    def assign_points_to_centroids(self):
        """
        Method assigns data points to the closest centroids.
        """
        # Calculating distances from all centroids to data points
        all_centroids_distances: list[NDArray] = [
            (
                self.euclidean_distance(
                    data1=centroid,
                    data2=self.data
                )
            ) for centroid in self.clusters.keys()
        ]
        
        closest_centroids_indicies: NDArray[np.int64] = (
            np.argmin(a=all_centroids_distances, axis=0)
        )
        
        for ind, centroid_ind in enumerate(closest_centroids_indicies):
            closest_centroid: tuple[np.float64] = (
                list(self.clusters.keys())[centroid_ind]
            )
            self.clusters[closest_centroid].append(self.data[ind])
        
    def get_inertia(self) -> np.float64:
        """
        Method calculates interia for final clusters.
        Returns inertia as a float.
        """
        # If no cluster points' list is not empty - fit_model was not implemented.
        empty_clusters: bool = True
        for points in self.clusters.values():
            debug_logger.debug(
                msg=f"points:\n{points}"
            )
            if points != []:
                empty_clusters = False

        if empty_clusters:    
            error_logger.warning(msg="There are empty clusters.")
        
        inertia: float = 0.0
        for ind, points in enumerate(self.clusters.values()):
            if points != []:
                centroid: NDArray[np.float64] = (
                    list(self.clusters.keys())[ind]
                )
                inertia += np.sum((centroid - np.asarray(points))**2)
                     
        return inertia
    
    def get_best_result(self, runs: int) -> dict[
        tuple[np.float64], list[NDArray[np.float64]]
    ]:
        """
        Method implements defined runs to get a result with min inertia.
        Returns a dictionery with clusters of the best result.
        """
        results: dict[np.float64, dict[tuple[np.float64], list[NDArray]]] = {}
        for _ in range(runs):
            self.kmeans_plusplus()
            self.assign_points_to_centroids()
            inertia: np.float64 = self.get_inertia()
            results[inertia] = self.clusters
            self.clusters = {}
            
        best_inertia: np.float64 = min(results.keys())
        best_result: dict[
            tuple[np.float64], list[NDArray[np.float64]]
        ] = results[best_inertia]
        
        return best_result


### FUNCTIONS

def get_kmeans_elbow_data(
        data: pd.DataFrame, k_variants: list[int]
    ) -> dict[int, np.float64]:
        
        elbow_data: dict[int, np.float64] = {}
        for k in k_variants:
            kmeans: KMeans = KMeans(data=data, k=k)
            kmeans.kmeans_plusplus()
            kmeans.assign_points_to_centroids()
            
            elbow_data[k] = kmeans.get_inertia()
            
        return elbow_data


### IMPLEMENTATION

main_data: pd.DataFrame = pd.read_csv(filepath_or_buffer="assets/data/for_kmeans.csv")

# 1st combination

data1_for_kmeans: pd.DataFrame = main_data[
    ["current_funding_level(num)", "startup_age"]
]

k_variants1: list[int] = list(range(1, 6))
elbow_data1: dict[int, np.float64] = (
    get_kmeans_elbow_data(data=data1_for_kmeans, k_variants=k_variants1)
)
fig, ax = plt.subplots()
sns.lineplot(data=elbow_data1)
plt.savefig("assets/visualizations/kmeans/elbow1.png")
# Optimal k is 3
data1_k: int = 3
plt.close()
      
kmeans_data1: KMeans = KMeans(data=data1_for_kmeans, k=data1_k) 
data1_kmeans_best_res: dict[tuple[np.float64], list[NDArray[np.float64]]] = (
    kmeans_data1.get_best_result(runs=10)
)
    
colors1: list[str] = ["red", "green", "orange"]
data1_kmeans_centers: list[list[NDArray]] = list(data1_kmeans_best_res.keys())
data1_kmeans_points: list[tuple] = list(data1_kmeans_best_res.values())

fig, ax = plt.subplots(figsize=(7, 7))
for i in range(data1_k):
    center: NDArray = np.array(object=data1_kmeans_centers[i])
    points: NDArray = np.array(object=data1_kmeans_points[i])
    plt.scatter(
       x=points[:,0],
       y=points[:,1],
       c=colors1[i],
       s=20.0
    )
    plt.scatter(
        x=center[0],
        y=center[1],
        c=colors1[i],
        marker="x",
        s=40.0
    )
plt.savefig("assets/visualizations/kmeans/data1.png")
plt.show()

all_clusters_points1: list[list[NDArray]] = list(
    data1_kmeans_best_res.values()
)
main_data["kmeans1_cluster"] = np.nan
for i, row in main_data.iterrows():
    data_point: NDArray = np.array(
        object=[row["current_funding_level(num)"], row["startup_age"]]
    )
    if any(
        np.array_equal(
            a1=data_point,
            a2=cluster_point) 
        for cluster_point in all_clusters_points1[0]
    ):
        main_data.at[i, "kmeans1_cluster"] = 0
    elif any(
        np.array_equal(
            a1=data_point,
            a2=cluster_point) 
        for cluster_point in all_clusters_points1[1]
    ):
        main_data.at[i, "kmeans1_cluster"] = 1
    elif any(
        np.array_equal(
            a1=data_point,
            a2=cluster_point) 
        for cluster_point in all_clusters_points1[2]
    ):
        main_data.at[i, "kmeans1_cluster"] = 2
    

# 2nd combination

data2_for_kmeans: pd.DataFrame = main_data[
    ["startup_age", "amount_raised_log"]
]


k_variants2: list[int] = list(range(1, 6))
elbow_data2: dict[int, np.float64] = (
    get_kmeans_elbow_data(data=data2_for_kmeans, k_variants=k_variants2)
)
fig, ax = plt.subplots()
sns.lineplot(data=elbow_data2)
plt.savefig("assets/visualizations/kmeans/elbow2.png")
# Optimal k is 3
data2_k: int = 3
plt.close()

kmeans_data2: KMeans = KMeans(data=data2_for_kmeans, k=data2_k) 
data2_kmeans_best_res: dict[tuple[np.float64], list[NDArray[np.float64]]] = (
    kmeans_data2.get_best_result(runs=10)
)
    
colors2: list[str] = ["red", "green", "orange"]
data2_kmeans_centers: list[list[NDArray]] = list(data2_kmeans_best_res.keys())
data2_kmeans_points: list[tuple] = list(data2_kmeans_best_res.values())

fig, ax = plt.subplots(figsize=(15, 15))
for i in range(data2_k):
    center: NDArray = np.array(object=data2_kmeans_centers[i])
    points: NDArray = np.array(object=data2_kmeans_points[i])
    plt.scatter(
       x=points[:,0],
       y=points[:,1],
       c=colors2[i],
       s=20.0
    )
    plt.scatter(
        x=center[0],
        y=center[1],
        c=colors2[i],
        marker="x",
        s=40.0
    )
plt.savefig("assets/visualizations/kmeans/data2.png")
plt.show()

all_clusters_points2: list[list[NDArray]] = list(
    data2_kmeans_best_res.values()
)
main_data["kmeans2_cluster"] = np.nan
for i, row in main_data.iterrows():
    data_point: NDArray = np.array(
        object=[row["startup_age"], row["amount_raised_log"]]
    )
    if any(
        np.array_equal(
            a1=data_point,
            a2=cluster_point) 
        for cluster_point in all_clusters_points2[0]
    ):
        main_data.at[i, "kmeans2_cluster"] = 0
    elif any(
        np.array_equal(
            a1=data_point,
            a2=cluster_point) 
        for cluster_point in all_clusters_points2[1]
    ):
        main_data.at[i, "kmeans2_cluster"] = 1
    elif any(
        np.array_equal(
            a1=data_point,
            a2=cluster_point) 
        for cluster_point in all_clusters_points2[2]
    ):
        main_data.at[i, "kmeans2_cluster"] = 2


# 3rd combination        
        
data3_for_kmeans: pd.DataFrame = main_data[
    ["current_funding_level(num)", "startup_age", "amount_raised_log"]
]

k_variants3: list[int] = list(range(1, 6))
elbow_data3: dict[int, np.float64] = (
    get_kmeans_elbow_data(data=data3_for_kmeans, k_variants=k_variants3)
)
fig, ax = plt.subplots()
sns.lineplot(data=elbow_data3)
plt.savefig("assets/visualizations/kmeans/elbow3.png")
# Optimal k is 3
data3_k: int = 3
plt.close()

kmeans_data2: KMeans = KMeans(data=data3_for_kmeans, k=data3_k) 
data3_kmeans_best_res: dict[tuple[np.float64], list[NDArray[np.float64]]] = (
    kmeans_data2.get_best_result(runs=10)
)
    
colors3: list[str] = ["red", "green", "orange"]
data3_kmeans_centers: list[list[NDArray]] = list(data3_kmeans_best_res.keys())
data3_kmeans_points: list[tuple] = list(data3_kmeans_best_res.values())

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")
for i in range(data3_k):
    center: NDArray = np.array(object=data3_kmeans_centers[i])
    points: NDArray = np.array(object=data3_kmeans_points[i])
    ax.scatter(
       xs=points[:,0],
       ys=points[:,1],
       zs=points[:,2],
       c=colors3[i],
       s=20.0
    )
    ax.scatter(
        xs=center[0],
        ys=center[1],
        zs=center[2],
        c=colors3[i],
        marker="x",
        s=40.0
    )
ax.set_xlabel("current_funding_level(num)")
ax.set_ylabel("startup_age")
ax.set_zlabel("amount_raised_log")
plt.savefig("assets/visualizations/kmeans/data3.png")
plt.show()

all_clusters_points3: list[list[NDArray]] = list(
    data3_kmeans_best_res.values()
)
main_data["kmeans3_cluster"] = np.nan
for i, row in main_data.iterrows():
    data_point: NDArray = np.array(
        object=[
            row["current_funding_level(num)"],
            row["startup_age"],
            row["amount_raised_log"]
        ]
    )
    if any(
        np.array_equal(
            a1=data_point,
            a2=cluster_point) 
        for cluster_point in all_clusters_points3[0]
    ):
        main_data.at[i, "kmeans3_cluster"] = 0
    elif any(
        np.array_equal(
            a1=data_point,
            a2=cluster_point) 
        for cluster_point in all_clusters_points3[1]
    ):
        main_data.at[i, "kmeans3_cluster"] = 1
    elif any(
        np.array_equal(
            a1=data_point,
            a2=cluster_point) 
        for cluster_point in all_clusters_points3[2]
    ):
        main_data.at[i, "kmeans3_cluster"] = 2
