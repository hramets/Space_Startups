import logging
import pandas as pd
import numpy as np
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
        random_centorid: NDArray[np.int64] = (
            self.data[np.random.randint(low=0, high=len(self.data))]
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
            closest_centroid: NDArray[np.float64] = (
                list(self.clusters.keys())[centroid_ind]
            )
            self.clusters[closest_centroid].append(self.data[ind])
        
        debug_logger.debug(msg=f"clusters:\n{self.clusters}")
        
    def get_inertia(self) -> np.float64:
        """
        Method calculates interia for final clusters.
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
    
    def get_best_result(self, runs: int):
        results: dict[np.float64, dict[NDArray, list[NDArray]]] = {}
        for _ in range(runs):
            self.kmeans_plusplus()
            self.assign_points_to_centroids()
            inertia: np.float64 = self.get_inertia()
            results[inertia] = self.clusters
            
        best_inertia: np.float64 = min(results.keys())
        best_result: dict[NDArray, NDArray] = results[best_inertia]
        
        return best_result
