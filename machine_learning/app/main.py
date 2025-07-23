import os
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import litserve as ls
from abc import ABC, abstractmethod
import pandas as pd
from ingestion_pipeline import factory
from typing import Dict, Any


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

class UnsupervisedLearningStrategy(ABC):
    @abstractmethod
    def execute(self, data: pd.DataFrame) -> Dict[str, Any]:
        pass

class KMeansStrategy(UnsupervisedLearningStrategy):
    def execute(self, data: pd.DataFrame) -> Dict[str, Any]:
        # Optimize number of clusters using elbow method
        wcss = []
        for i in range(2, 11):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)
        
        # Find optimal number of clusters using silhouette score
        best_score = -1
        best_n_clusters = 2
        for n in range(2, 11):
            kmeans = KMeans(n_clusters=n, random_state=42)
            labels = kmeans.fit_predict(data)
            score = silhouette_score(data, labels)
            if score > best_score:
                best_score = score
                best_n_clusters = n

        # Final clustering with optimal number of clusters
        kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
        clusters = kmeans.fit_predict(data)
        
        # Calculate metrics
        metrics = {
            'silhouette_score': silhouette_score(data, clusters),
            'calinski_harabasz_score': calinski_harabasz_score(data, clusters)
        }
        
        return {
            'clusters': clusters.tolist(),
            'metrics': metrics,
            'optimal_clusters': best_n_clusters
        }

class DBSCANStrategy(UnsupervisedLearningStrategy):
    def execute(self, data: pd.DataFrame) -> Dict[str, Any]:
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(data)
        
        # Calculate metrics (excluding noise points)
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[dbscan.core_sample_indices_] = True
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters > 1:
            metrics = {
                'silhouette_score': silhouette_score(data[labels != -1], labels[labels != -1]),
                'n_clusters': n_clusters
            }
        else:
            metrics = {'n_clusters': n_clusters}
        
        return {
            'clusters': labels.tolist(),
            'metrics': metrics
        }

class AgglomerativeStrategy(UnsupervisedLearningStrategy):
    def execute(self, data: pd.DataFrame) -> Dict[str, Any]:
        agglomerative = AgglomerativeClustering(n_clusters=None, distance_threshold=0)
        clusters = agglomerative.fit_predict(data)
        
        metrics = {
            'silhouette_score': silhouette_score(data, clusters),
            'calinski_harabasz_score': calinski_harabasz_score(data, clusters)
        }
        
        return {
            'clusters': clusters.tolist(),
            'metrics': metrics
        }


class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        self.strategies = {
            "kmeans": KMeansStrategy(),
            "dbscan": DBSCANStrategy(),
            "agglomerative": AgglomerativeStrategy()
        }
        self.selected_strategy = self.strategies["kmeans"] 
        self.ingestion_factory = factory

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data for clustering"""
        # Handle missing values
        data = data.fillna(data.mean())
        
        # Convert categorical variables
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
        
        # Scale numerical features
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        
        return data

    def decode_request(self, request):
        return request["input"], request.get("strategy", "kmeans"), request.get("pipeline", "csv"), request.get("file_path", "")

    def predict(self, x):
        data, strategy_key, pipeline_key, file_path = x
        print(f"Data: {data}, Strategy: {strategy_key}, Pipeline: {pipeline_key}, File Path: {file_path}")
        try:
            file_path = os.path.join(project_root, file_path)
            data = pd.read_csv(file_path)
            
            # Preprocess the data
            data = self.preprocess_data(data)
            
            self.selected_strategy = self.strategies.get(strategy_key, self.strategies["kmeans"])
            result = self.selected_strategy.execute(data)
            
            # Use the ingestion pipeline
            pipeline = self.ingestion_factory.get_pipeline(pipeline_key, file_path=file_path)
            ingestion_result = pipeline.process()

            return {
                "output": result,
                "ingestion": ingestion_result,
                "status": "success"
            }
        except Exception as e:
            print(f"Error processing data: {e}")
            return {"error": str(e)}

    def encode_response(self, output):
        return {"output": output}

if __name__ == "__main__":
    api = SimpleLitAPI()
    server = ls.LitServer(api, accelerator="cpu")
    server.run(port=8000)



