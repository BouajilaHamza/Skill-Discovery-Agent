# Skill Discovery Agent with DIAYN

An implementation of the Diversity is All You Need (DIAYN) algorithm for skill discovery in MiniGrid environments. This project demonstrates how agents can learn diverse skills in an unsupervised manner using information-theoretic objectives.

## Features

- **Unsupervised Skill Discovery**: Learns diverse skills without external rewards
- **MiniGrid Integration**: Works with various MiniGrid environments
- **PyTorch Lightning**: Clean and modular implementation using PyTorch Lightning
- **Visualization Tools**: Tools to visualize and analyze learned skills
- **Configurable**: Easily configurable hyperparameters and environment settings

## Project Structure

```
.
├── configs/                    # Configuration files
│   └── diayn.yaml             # DIAYN agent configuration
├── src/                       # Source code
│   ├── agents/                # Agent implementations
│   │   ├── base_agent.py      # Base agent class
│   │   └── diayn_agent.py     # DIAYN agent implementation
│   ├── envs/                  # Environment wrappers
│   │   ├── __init__.py        # Environment registration
│   │   └── minigrid_wrapper.py# MiniGrid environment wrapper
│   ├── models/                # Model architectures
│   │   └── base_model.py      # Base model class
│   └── scripts/               # Training and evaluation scripts
│       ├── train.py           # Training script
│       └── visualize_skills.py # Skill visualization
├── tests/                     # Test files
│   └── __init__.py
├── pyproject.toml             # Python project configuration
└── README.md                  # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/skill-discovery-agent.git
   cd skill-discovery-agent
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

### Running with Docker

```bash
docker-compose up --build
```

## Model Training

Before using the organizer, you'll need to train the machine learning models. The project includes two main types of models:

1. **Text Clustering**: For organizing text-based files (TXT, PDF, DOCX, etc.)
2. **Image Clustering**: For organizing image files (JPG, PNG, etc.)

### Prerequisites

1. Install the required dependencies:
   ```bash
   pip install -e .
   ```

2. For image processing, you'll also need:
   ```bash
   pip install torch torchvision pillow
   ```

### Preparing Training Data

Organize your training data in the following structure:

```
training_data/
├── text/
│   ├── category1/
│   │   ├── doc1.txt
│   │   └── doc2.txt
│   └── category2/
│       └── doc3.txt
└── images/
    ├── categoryA/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── categoryB/
        └── img3.jpg
```

### Training Text Clustering Model

```bash
# Basic usage
python -m machine_learning.models text \
  --data_path /path/to/training_data/text \
  --output_dir models/text_model

# Advanced options
python -m machine_learning.models text \
  --data_path /path/to/training_data/text \
  --output_dir models/text_model \
  --model_type kmeans \
  --n_clusters 5 \
  --n_components 100
```

### Training Image Clustering Model

```bash
# Basic usage
python -m machine_learning.models image \
  --data_path /path/to/training_data/images \
  --output_dir models/image_model

# Advanced options
python -m machine_learning.models image \
  --data_path /path/to/training_data/images \
  --output_dir models/image_model \
  --model_type kmeans \
  --n_clusters 5 \
  --feature_type deep \
  --n_components 128
```

### Training Options

#### Text Clustering Options:
- `--model_type`: Clustering algorithm (`kmeans`, `dbscan`, `agglomerative`, `gmm`, `optics`)
- `--n_clusters`: Number of clusters (default: 5, not used for DBSCAN/OPTICS)
- `--n_components`: Number of components for dimensionality reduction (default: 100)
- `--no_dim_reduction`: Disable dimensionality reduction
- `--random_state`: Random seed (default: 42)

#### Image Clustering Options:
- `--model_type`: Clustering algorithm (`kmeans`, `dbscan`, `gmm`, `optics`)
- `--n_clusters`: Number of clusters (default: 5, not used for DBSCAN/OPTICS)
- `--feature_type`: Type of features to use (`deep`, `color`, `texture`, `all`)
- `--n_components`: Number of components for dimensionality reduction (default: 128)
- `--no_dim_reduction`: Disable dimensionality reduction
- `--random_state`: Random seed (default: 42)

### Model Output

After training, each model will be saved with the following structure:

```
output_dir/
├── model.pkl          # Trained model
└── metadata.json      # Model metadata and parameters
```

The `metadata.json` file contains information about the training process and model configuration.

## Usage

After training the models, you can run the client application:

```bash
python client.py --source /path/to/files --destination /path/to/organized/files --model_dir /path/to/trained/models
```

## Configuration

Edit `config/organizer_config.py` to customize the organization rules and model parameters.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




Research Paper :
https://arxiv.org/pdf/1802.06070