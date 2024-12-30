import pandas as pd
import numpy as np
from numpy.typing import NDArray


class KMeans:
    
    def __init__(self, data: pd.DataFrame, k: int):
        self.data: pd.DataFrame = data
        self.clusters: dict[int, dict[str, list | NDArray]] = {}
        self.k = k
    
    @staticmethod
    def euclidean_distance(
        p1: NDArray[np.float64 | np.int64],
        p2: NDArray[np.float64 | np.int64]
    ) -> np.float64:
        """
        Method calculate Euclidean distance between two points.
        """
        return np.sqrt(np.sum((p1 - p2)**2))
      
    def initialize_centroids(self):
        """
        Method defines random initial centroids.
        Then it initializes storages for clusters' data and
        adds centroids to it.
        """
        if self.data.empty:
            raise ValueError("Data cannot be empty.")
        if not self.k:
            raise ValueError("k must be defined.")
        
        # Creating random centroids within the data scale.
        centroids: NDArray[np.float64] = np.random.uniform(
            low=min(np.amin(a=self.data, axis=0)),
            high=max(np.amax(a=self.data, axis=0)),
            size=(self.k, self.data.shape[1])
        )

        for ind, centroid in enumerate(centroids):
            self.clusters[ind] = {}
            self.clusters[ind]["centroid"] = centroid
            self.clusters[ind]["points"] =  []

    def assign_clusters(self) -> None:
        """
        Method assigns data points to the centorids.
        It finds the smallest distance between points and centroids.
        Adds points to the corresponding clusters.
        """
        if self.clusters == {}:
            raise KeyError("Centroids are not initialized.")

        data: NDArray[np.float64] = np.asarray(self.data) 
        for _, data_point in enumerate(data):
            distances: list[np.float64] = []

            for cluster in range(self.k):
                centroid_point: NDArray[np.float64 | np.int64] = (
                    self.clusters[cluster]["centroid"]
                )
                distances.append(
                    self.euclidean_distance(p1=centroid_point, p2=data_point)
                )

            min_dist_centroid: np.int64 = (
                distances.index(min(distances))
            )
            self.clusters[min_dist_centroid]["points"].append(data_point)

    def update_clusters(self) -> None:
        """
        Method updates clusters. It redefines centroids by
        calculating data points' mean. 
        Removes data points from dictionary with clusters.
        """
        for cluster in range(self.k):
            new_centroid_point: NDArray[np.float64] = (
                np.mean(a=np.asarray(self.clusters[cluster]["points"]), axis=0)
            )
            self.clusters[cluster]["centroid"] = new_centroid_point
            self.clusters[cluster]["points"] = []
     
    @staticmethod       
    def stop_training(
        old_centroids: NDArray[np.float64],
        new_centroids: NDArray[np.float64],
        stop_diff: float
    ) -> bool:
        """
        Method calculates a difference between old and new centroid points.
        If the difference is equal or smaller than the defined one -
        return True, otherwise False.
        """
        stop: bool = False
        if np.amax(np.sqrt((old_centroids - new_centroids)**2)) <= stop_diff:
            stop = True
            
        return stop
    
    def fit_model(self) -> None:
        """
        Method fits k-mean clustering model. Uses the methods from class.
        As result - clusters instance with final clusters data.
        """
        self.initialize_centroids()

        stop: bool = False
        while not stop:
            old_centroids: list[NDArray[np.int64 | np.float64]] = [
                self.clusters[cluster]["centroid"] for cluster in range(self.k)
            ]
            
            self.assign_clusters()
            self.update_clusters()
            
            new_centroids: list[NDArray[np.float64]] = [
                self.clusters[cluster]["centroid"] for cluster in range(self.k)
            ]
            stop = self.stop_training(
                old_centroids=np.asarray(old_centroids),
                new_centroids=np.asarray(new_centroids),
                stop_diff=0.0001
            )
