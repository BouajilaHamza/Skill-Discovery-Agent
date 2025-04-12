from ingestion_pipeline import __version__, factory, CSVIngestionPipeline


def test_version():
    assert __version__ == '0.1.0'

def test_register_and_get_pipeline():
    # Register a mock pipeline
    class MockPipeline:
        def __init__(self, param):
            self.param = param

        def process(self):
            return f"Mock processing with {self.param}"

    factory.register_pipeline("mock", MockPipeline)

    # Retrieve and test the pipeline
    pipeline = factory.get_pipeline("mock", param="test_param")
    assert isinstance(pipeline, MockPipeline)
    assert pipeline.process() == "Mock processing with test_param"

def test_csv_ingestion_pipeline():
    # Test the CSV ingestion pipeline
    pipeline = CSVIngestionPipeline(file_path="test.csv")
    result = pipeline.process()
    assert result == "Processing CSV file at test.csv"
