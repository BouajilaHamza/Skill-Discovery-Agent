# Intelligent File Organizer

An intelligent file organization system that automatically categorizes and organizes files using machine learning.

## Features

- **Smart Categorization**: Automatically categorizes files based on content analysis
- **Multi-format Support**: Handles various file types including documents and images
- **Customizable Rules**: Define your own organization rules and preferences
- **Machine Learning**: Utilizes clustering algorithms for intelligent file grouping
- **Docker Support**: Easy deployment with Docker containers

## Project Structure

```
.
├── config/                 # Configuration files
├── data_engineering/      # Data processing and feature extraction
├── machine_learning/      # ML models and application code
│   ├── app/               # FastAPI application
│   ├── models/            # ML model implementations
│   ├── research/          # Jupyter notebooks for research
│   └── tests/             # Test files
├── client.py              # Client application
├── compose.yaml           # Docker Compose configuration
├── pyproject.toml         # Python project configuration
└── uv.lock                # Dependency lock file
```

## Getting Started

### Prerequisites

- Python 3.12+
- Docker and Docker Compose (for containerized deployment)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Intelligent-File-Organizer.git
   cd Intelligent-File-Organizer
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

### Running with Docker

```bash
docker-compose up --build
```

## Usage

Run the client application:

```bash
python client.py --source /path/to/files --destination /path/to/organized/files
```

## Configuration

Edit `config/organizer_config.py` to customize the organization rules and model parameters.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

