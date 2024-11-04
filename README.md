# Deep Belief Network (DBN) Implementation

This repository contains an implementation of Deep Belief Networks (DBN) using Theano, including Restricted Boltzmann Machines (RBM) and Multi-Layer Perceptron (MLP) components. The implementation is designed to work with the MNIST dataset for handwritten digit recognition.

[![Python Version](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/github/followers/EbGazar?label=Follow&style=social)](https://github.com/EbGazar)

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Details](#model-details)
- [Training](#training)
- [Results Visualization](#results-visualization)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Prerequisites

- Python 3.6 (specific version required)
- CUDA (optional, for GPU support)
- Conda package manager

## Installation

1. First, create and activate a new conda environment:
```bash
# Create new environment
conda create -n dbn python=3.6

# Activate environment
conda activate dbn
```

2. Clone the repository:
```bash
git clone https://github.com/EbGazar/deep-belief-network.git
cd deep-belief-network
```

3. Install the required packages:
```bash
# Using pip
pip install -r requirements.txt

# OR using conda
conda env create -f environment.yml
```

## Project Structure

```
deep-belief-network/
├── __init__.py
├── DBN.py          # Deep Belief Network implementation
├── logistic_sgd.py # Logistic Regression with SGD
├── mlp.py          # Multi-Layer Perceptron
├── rbm.py          # Restricted Boltzmann Machine
├── utils.py        # Utility functions
├── mnist.pkl.gz    # MNIST dataset
├── requirements.txt    # Package dependencies
├── environment.yml     # Conda environment file
└── setup.py           # Setup configuration
```

## Usage

### Basic Usage

To train the DBN on MNIST dataset:
```bash
python src/DBN.py
```

### Advanced Usage

To modify the training parameters:

```python
from src.DBN import test_DBN

test_DBN(finetune_lr=0.1,             # Learning rate for fine-tuning
         pretraining_epochs=100,       # Number of pretraining epochs
         pretrain_lr=0.01,            # Pretraining learning rate
         k=1,                         # Number of Gibbs steps
         training_epochs=1000,        # Number of training epochs
         dataset='mnist.pkl.gz',      # Dataset file
         batch_size=10)               # Size of each batch
```

## Model Details

The DBN implementation consists of several key components:

1. **Deep Belief Network (DBN)**
   - Stacked RBMs with fine-tuning
   - Configurable number of hidden layers
   - Supervised fine-tuning using backpropagation

2. **Restricted Boltzmann Machine (RBM)**
   - Unsupervised learning component
   - Contrastive Divergence training
   - Binary hidden and visible units

3. **Multi-Layer Perceptron (MLP)**
   - Used in the fine-tuning phase
   - Configurable hidden layers
   - Supports various activation functions

## Training

The training process consists of two phases:

1. **Pretraining Phase**
   - Layer-wise training of RBMs
   - Unsupervised learning
   - Parameters:
     - pretraining_epochs: 100
     - pretrain_lr: 0.01

2. **Fine-tuning Phase**
   - Supervised training using backpropagation
   - Parameters:
     - training_epochs: 1000
     - finetune_lr: 0.1

## Results Visualization

The training process will output:
- Training error rates
- Validation error rates
- Test set performance
- Training time statistics

Visual outputs include:
- RBM filters visualization
- Training progress plots
- Reconstruction quality assessment

## Contributing

1. Fork the repository
2. Create your feature branch:
```bash
git checkout -b feature/your-feature-name
```
3. Commit your changes:
```bash
git commit -am 'Add some feature'
```
4. Push to the branch:
```bash
git push origin feature/your-feature-name
```
5. Submit a pull request

## Troubleshooting

Common issues and solutions:

1. **Python Version Mismatch**
   ```bash
   # Verify Python version
   python --version
   # Should output Python 3.6.x
   ```

2. **Theano Installation Issues**
   ```bash
   # Try installing Theano separately
   pip install Theano==1.0.4
   ```

3. **MNIST Dataset Download Fails**
   - Manually download from [MNIST website](http://yann.lecun.com/exdb/mnist/)
   - Place in the `data` directory

4. **GPU Issues**
   - Check CUDA installation
   - Verify Theano GPU configuration in `.theanorc`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{deep-belief-network,
  author = {EbGazar},
  title = {Deep Belief Network Implementation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/EbGazar/deep-belief-network}
}
```

## Acknowledgments

- Deep Learning Tutorials
- Theano Development Team
- MNIST Dataset creators

## Contact

For questions and support, please open an issue on the [GitHub repository](https://github.com/EbGazar/Deep-Belief-Network-Implementation/issues).
