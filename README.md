<div align="center">

# 🎯 Skill Discovery with DIAYN

[![arXiv](https://img.shields.io/badge/arXiv-1802.06070-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/1802.06070)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow.svg?style=flat-square)](https://huggingface.co/your-username)

</div>

An efficient PyTorch implementation of the **Diversity is All You Need (DIAYN)** algorithm for unsupervised skill discovery in reinforcement learning. This project enables agents to autonomously learn diverse behaviors in MiniGrid environments using information-theoretic objectives.

## 📝 Paper Reference

> [**Diversity is All You Need: Learning Skills without a Reward Function**](https://arxiv.org/abs/1802.06070)  
> Benjamin Eysenbach, Abhishek Gupta, Julian Ibarz, Sergey Levine  
> *International Conference on Learning Representations (ICLR), 2019*

## ✨ Features

- **🤖 Unsupervised Skill Discovery**: Learn diverse skills without external rewards
- **🔄 MiniGrid Integration**: Test in various MiniGrid environments
- **⚡ PyTorch Implementation**: Optimized for performance and readability
- **📊 Visualization Tools**: Built-in visualization of learned skills and metrics
- **🔧 Configurable**: Easy hyperparameter tuning via YAML configs
- **📈 TensorBoard Logging**: Track training progress and metrics
- **🤗 HuggingFace Ready**: Easy model sharing and loading

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/BouajilaHamza/Skill-Discovery-Agent.git
   cd Skill-Discovery-Agent
   ```

2. **Install dependencies**:
   ```bash
   # Using pip
   pip install -e .
   
   # Or using UV (faster)
   uv sync
   ```

### Training

Train a new DIAYN agent with default settings:

```bash
uv run src/scripts/train.py --config configs/diayn.yaml
```

### Visualizing Learned Skills

Visualize the skills learned by a trained agent:

```bash
uv run src/scripts/visualize_skills.py --checkpoint /path/to/checkpoint.pt
```

## 🏗️ Project Structure

```
.
├── configs/                    # Configuration files
│   └── diayn.yaml             # DIAYN agent configuration
├── logs/                      # Training logs and checkpoints
├── src/                       # Source code
│   ├── agents/                # Agent implementations
│   │   ├── base_agent.py      # Base agent class
│   │   └── diayn_agent.py     # DIAYN agent implementation
│   ├── envs/                  # Environment wrappers
│   │   ├── __init__.py        # Environment registration
│   │   └── minigrid_wrapper.py# MiniGrid environment wrapper
│   ├── models/                # Model architectures
│   │   ├── base_model.py      # Base model class
│   │   ├── encoder.py         # State encoder
│   │   └── discriminator.py   # Skill discriminator
│   └── scripts/               # Training and evaluation scripts
│       ├── train.py           # Training script
│       ├── visualize_skills.py # Skill visualization
│       └── visualization.py   # Training metrics visualization
├── tests/                     # Unit tests
├── .gitignore                 # Git ignore file
├── pyproject.toml             # Project configuration
└── README.md                  # This file
```

## 🤗 Model Sharing

### Uploading to Hugging Face

To share your trained models on Hugging Face Hub:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="logs/your_experiment/checkpoints",
    repo_id="your-username/diayn-minigrid",
    repo_type="model",
)
```

### Loading from Hugging Face

```python
from huggingface_hub import hf_hub_download
import torch

checkpoint_path = hf_hub_download(
    repo_id="your-username/diayn-minigrid",
    filename="diayn_final.pt"
)
model = torch.load(checkpoint_path, map_location='cpu')
```

## 📊 Results

### Training Metrics

![Training Metrics](assets/training_metrics.png)

### Learned Skills

![Learned Skills](assets/learned_skills.gif)

## 📚 Documentation

### Configuration

Key configuration parameters in `configs/diayn.yaml`:

```yaml
# Environment
env_id: "MiniGrid-Empty-8x8-v0"
obs_type: "rgb"

# Training
num_skills: 8
batch_size: 64
learning_rate: 3e-4
gamma: 0.99
entropy_coeff: 0.01
replay_size: 10000
max_episodes: 1000
```

### Available Commands

| Command | Description |
|---------|-------------|
| `python -m src.scripts.train` | Train a new DIAYN agent |
| `python -m src.scripts.visualize_skills` | Visualize learned skills |
| `python -m src.scripts.visualization` | Generate training plots |
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

## 🛠️ Usage

### Training with Custom Configuration

```bash
python -m src.scripts.train \
  --config configs/diayn.yaml \
  --num_skills 16 \
  --env_id "MiniGrid-Empty-16x16-v0"
```

### Monitoring Training

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir=logs/
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📄 Citation

If you use this code in your research, please cite the original DIAYN paper:

```bibtex
@inproceedings{eysenbach2018diversity,
  title={Diversity is All You Need: Learning Skills without a Reward Function},
  author={Eysenbach, Benjamin and Gupta, Abhishek and Ibarz, Julian and Levine, Sergey},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2019}
}
```

## 🙏 Acknowledgements

- [DIAYN Paper](https://arxiv.org/abs/1802.06070) for the original algorithm
- [MiniGrid](https://github.com/Farama-Foundation/MiniGrid) for the environment
- [PyTorch](https://pytorch.org/) for the deep learning framework

```bash
python client.py --source /path/to/files --destination /path/to/organized/files --model_dir /path/to/trained/models
```

## Configuration

Edit `config/organizer_config.py` to customize the organization rules and model parameters.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.