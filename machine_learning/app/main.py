import sys
import os
from sklearn.cluster import KMeans, DBSCAN
# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

import litserve as ls
from abc import ABC, abstractmethod
from ingestion_pipeline import factory

# Define the Strategy base class
class UnsupervisedLearningStrategy(ABC):
    @abstractmethod
    def execute(self, data):
        pass

# Concrete Strategy: K-Means
class KMeansStrategy(UnsupervisedLearningStrategy):
    def execute(self, data):
        # Placeholder for K-Means logic
        # Example: KMeans clustering
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(data)
        labels = kmeans.labels_
        # Return the labels as the result
        return f"K-Means result for {labels.tolist()}"

# Concrete Strategy: DBSCAN
class DBSCANStrategy(UnsupervisedLearningStrategy):
    def execute(self, data):
        # Placeholder for DBSCAN logic
        # Example: DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan.fit(data)
        labels = dbscan.labels_
        # Return the labels as the result
        return f"DBSCAN result for {labels.tolist()}"

# Modify the SimpleLitAPI class to use the ingestion pipeline factory
class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        self.strategies = {
            "kmeans": KMeansStrategy(),
            "dbscan": DBSCANStrategy()
        }
        self.selected_strategy = self.strategies["kmeans"]  # Default strategy

        # Initialize the ingestion pipeline factory
        self.ingestion_factory = factory

    def decode_request(self, request):
        return request["input"], request.get("strategy", "kmeans"), request.get("pipeline", "csv"), request.get("file_path", "")

    def predict(self, x):
        data, strategy_key, pipeline_key, file_path = x

        # Select the strategy
        self.selected_strategy = self.strategies.get(strategy_key, self.strategies["kmeans"])
        result = self.selected_strategy.execute(data)

        # Use the ingestion pipeline
        pipeline = self.ingestion_factory.get_pipeline(pipeline_key, file_path=file_path)
        ingestion_result = pipeline.process()

        return {"output": result, "ingestion": ingestion_result}

    def encode_response(self, output):
        return {"output": output}

if __name__ == "__main__":
    api = SimpleLitAPI()
    server = ls.LitServer(api, accelerator="cpu")
    server.run(port=8000)



