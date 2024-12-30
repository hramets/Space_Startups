import unittest
from unittest.mock import MagicMock
from implementation.data_analysis.kmeans_class import KMeans
import pandas as pd
import numpy as np
from numpy.typing import NDArray


class TestKMeansClass(unittest.TestCase):
    def setUp(self):
        k: int = 3
        dataframe: pd.DataFrame = pd.DataFrame(
            columns=['A', 'B', 'C'],
            data=np.random.randint(low=0, high=100, size=(100, 3))
        )
        self.kmeans: KMeans = KMeans(data=dataframe, k=k)

    def test_euclidean_distance(self):
            self.kmeans.initialize_centroids()
            random_centroid: NDArray[np.float64] = self.kmeans.clusters[
                np.random.randint(low=0, high=self.kmeans.k)
            ]["centroid"]
            data: NDArray = np.asarray(self.kmeans.data)
            
            positive_n: bool = True
            for point in data:
                result: np.float64 = self.kmeans.euclidean_distance(
                    p1=random_centroid, p2=point
                )
                if result < 0:
                    positive_n = False
                    break
    
            self.assertEqual(
                first=positive_n,
                second=True,
                msg="While calculating Euclidean distance a negative number was got."
            )
  
    def test_initialize_centroids(self):
        self.kmeans.initialize_centroids()
        
        correct_centroid_shape: bool = True
        for cluster in range(self.kmeans.k):
            centroid_point: NDArray[np.float64] = self.kmeans.clusters[cluster]["centroid"]
            if centroid_point.shape != (3,):
                correct_centroid_shape = False
                break
        
        self.assertTrue(
            first=correct_centroid_shape,
            second=True,
            msg="Incorrect centroids' shape"
        )

    def test_assign_clusters(self):
        self.kmeans.initialize_centroids()
        self.kmeans.assign_clusters()
        
        random_cluster: NDArray[np.float64] = np.random.randint(
            low=0, high=self.kmeans.k
        )
        data_point_shape: tuple[int] = (
            self.kmeans.clusters[random_cluster]["points"][
                np.random.randint(
                    low=0, high=len(
                        self.kmeans.clusters[random_cluster]["points"]
                    )
                )
            ].shape
        )
        
        self.assertEqual(
            first=data_point_shape,
            second=(3,),
            msg="Incorrect data points' shape"
        )
        
    def test_update_clusters(self):
        self.kmeans.initialize_centroids()
        self.kmeans.assign_clusters()
        
        old_centroids: list[NDArray] = (
            [
                self.kmeans.clusters
                [
                    cluster
                ]
                [
                    "centroid"
                ] for cluster in range(self.kmeans.k)
            ]
        )
        self.kmeans.update_clusters()
        new_centroids: list[NDArray] = (
            [
                self.kmeans.clusters
                [
                    cluster
                ]
                [
                    "centroid"
                ] for cluster in range(self.kmeans.k)
            ]
            
        )

        self.assertEqual(
            first=np.array_equal(a1=old_centroids, a2=new_centroids),
            second=False,
            msg="Centroids are not updated."
        )
        self.assertEqual(
            first=self.kmeans.clusters[
                np.random.randint(low=0, high=self.kmeans.k)
            ]["points"],
            second=[],
            msg="Points are not updated"
        )
        
    def test_stop_training(self):
        old_centroids: NDArray[np.int64] = np.asarray([10, 5, 1])
        new_centroids: NDArray[np.int64] = np.asarray([1, 2, 3])
        
        result_true: bool = self.kmeans.stop_training(
            old_centroids=old_centroids,
            new_centroids=new_centroids,
            stop_diff=9
            )
        result_false: bool = self.kmeans.stop_training(
            old_centroids=old_centroids,
            new_centroids=new_centroids,
            stop_diff=7
            )
        
        self.assertTrue(
            expr=result_true is True,
            msg="Function stop_training defines when to stop incorrectly."
        )
        self.assertTrue(
            expr=result_false is False,
            msg="Function stop_training defines when to continue training incorrectly."
        )
        
    def test_fit_model(self):
        self.kmeans.stop_training = MagicMock(
            side_effect=[False, False, False, True]
        )
        self.kmeans.fit_model()
        
        self.assertEqual(
            first=self.kmeans.stop_training.call_count,
            second=4,
            msg="Error"
        )


if __name__ == "__main__":
    unittest.main()