__version__ = '0.1.0'

# Factory class for data ingestion pipelines
class IngestionPipelineFactory:
    def __init__(self):
        self.pipelines = {}

    def register_pipeline(self, name, pipeline_class):
        self.pipelines[name] = pipeline_class

    def get_pipeline(self, name, **kwargs):
        pipeline_class = self.pipelines.get(name)
        if not pipeline_class:
            raise ValueError(f"Pipeline '{name}' is not registered.")
        return pipeline_class(**kwargs)

# Example pipeline: CSV ingestion
class CSVIngestionPipeline:
    def __init__(self, file_path):
        self.file_path = file_path

    def process(self):
        # Placeholder for CSV processing logic
        return f"Processing CSV file at {self.file_path}"

# Register the CSV pipeline
factory = IngestionPipelineFactory()
factory.register_pipeline("csv", CSVIngestionPipeline)
