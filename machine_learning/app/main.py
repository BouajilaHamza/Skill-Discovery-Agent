import sys
import os
from sklearn.cluster import KMeans, DBSCAN
import litserve as ls
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.preprocessing import LabelEncoder
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)
from ingestion_pipeline import factory

class UnsupervisedLearningStrategy(ABC):
    @abstractmethod
    def execute(self, data):
        pass

class KMeansStrategy(UnsupervisedLearningStrategy):
    def execute(self, data):
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(data)
        labels = kmeans.labels_
        return f"K-Means result for {labels.tolist()}"

class DBSCANStrategy(UnsupervisedLearningStrategy):
    def execute(self, data):
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan.fit(data)
        labels = dbscan.labels_
        return f"DBSCAN result for {labels.tolist()}"


class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        self.strategies = {
            "kmeans": KMeansStrategy(),
            "dbscan": DBSCANStrategy()
        }
        self.selected_strategy = self.strategies["kmeans"] 
        self.ingestion_factory = factory

    def decode_request(self, request):
        return request["input"], request.get("strategy", "kmeans"), request.get("pipeline", "csv"), request.get("file_path", "")

    def predict(self, x):
        data, strategy_key, pipeline_key, file_path = x
        print(f"Data: {data}, Strategy: {strategy_key}, Pipeline: {pipeline_key}, File Path: {file_path}")
        try:
            file_path = os.path.join(project_root, file_path)
            data = pd.read_csv(file_path)         
            label_encoder = LabelEncoder()
            data['Company_Label'] = label_encoder.fit_transform(data['Ticker'])
            data = data.drop(columns=['Ticker'])
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                data['Date'] = data['Date'].map(pd.Timestamp.toordinal)
            data = data.dropna()
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            print(os.getcwd())
            return {"error": "Failed to read CSV file"}
        print(data)
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



