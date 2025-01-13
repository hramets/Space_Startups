import unittest
from unittest.mock import MagicMock
from implementation.data_analysis.kmeans_class import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
            data: NDArray = np.asarray(self.kmeans.data)
            random_centroid: NDArray[np.float64] = (
                data[np.random.randint(low=0, high=len(data))]
                )
            
            distances: NDArray = self.kmeans.euclidean_distance(
                data1=random_centroid, data2=data
            )
            print(data.shape, distances)
            self.assertEqual(
                first=distances.shape,
                second=(len(data),),
                msg="Method does not count all data points' distances with centroid"
            )

    def test_kmeans_plusplus(self):
        self.kmeans.kmeans_plusplus()
        self.assertEqual(
            first=len(self.kmeans.clusters.keys()),
            second=self.kmeans.k,
            msg="Number of centroids is not equal to defined k."
        )

        centroids: list[NDArray[np.float64]] = list(self.kmeans.clusters.keys())
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
            
                self.assertFalse(
                    expr=np.array_equal(a1=centroids[i], a2=centroids[j]),
                    msg="Centorids have the same position."
                )
            
    def test_assign_points_to_centroids(self):
        self.kmeans.kmeans_plusplus()
        self.kmeans.assign_points_to_centroids()

        for centroid in self.kmeans.clusters.keys():
            print(self.kmeans.clusters[centroid])
            self.assertTrue(
                expr=all(self.kmeans.clusters[centroid]),
                msg="Here are empty clusters."
            )

    def test_get_inertia(self):
        self.kmeans.kmeans_plusplus()
        self.kmeans.assign_points_to_centroids()
        
        inertia: np.float64 = self.kmeans.get_inertia()
        
        self.assertNotEqual(
            first=inertia,
            second=0.0,
            msg="Inertia calculation is not implemented."
        )
        
    def test_get_best_result(self):
        self.kmeans.kmeans_plusplus()
        self.kmeans.assign_points_to_centroids()
        best_result: dict[NDArray, list[NDArray]] = (
            self.kmeans.get_best_result(10)
        )
        self.assertIsNot(
            expr1=best_result,
            expr2={},
            msg="Result is empty."
        )
        

if __name__ == "__main__":
    unittest.main()