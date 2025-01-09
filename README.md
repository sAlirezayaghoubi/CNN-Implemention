# CNN Settings Exploration

This repository contains a Python project that tests various configurations of Convolutional Neural Networks (CNNs) to identify the best settings for optimizing performance. The project includes custom implementations for tiled convolutions, unshared convolutions, and locally connected layers. By experimenting with different parameters such as pooling types, dropout rates, kernel flipping, and padding styles, the project aims to compare CNN performance across settings.

## Features

- Custom CNN architecture with flexible configurations:
  - Tiled Convolutions
  - Unshared Convolutions
  - Locally Connected Layers
  - Adjustable sparsity
  - Configurable padding styles (valid, same, full)
- Integration with CIFAR-10 dataset for training and evaluation.
- Calculation of model parameters and FLOPs using PyTorch Profiler.
- Experimentation with different pooling types (e.g., MaxPooling, FractionalMaxPooling).
- Easy-to-adjust experimental settings.

## Code Structure

1. **Data Preparation**: Prepares CIFAR-10 dataset with normalization.
2. **CustomCNN Class**: Implements a configurable CNN model with support for various experimental settings.
3. **Training and Evaluation**: Functions to train and evaluate the CNNs on the CIFAR-10 dataset.
4. **Experiment Settings**: Predefined settings to test different configurations.
5. **Metrics Calculation**: Calculate the number of parameters and FLOPs for each CNN model.

## Requirements

Ensure you have the required libraries installed to run the project. See `requirements.txt` for the complete list.

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the experiments manaually in .ipynb file oo convert it to python script and run it: 
   ```bash
   jupyter nbconvert --to script CNN.ipynb
   python CNN.py
   ```

## Experiment Settings

You can modify the experiment settings in the script to test:
- Different convolution configurations (number of layers, kernel sizes, strides, paddings).
- Pooling types (e.g., MaxPooling, FractionalMaxPooling).
- Sparsity levels.
- Dropout rates.
- Kernel flipping.
- Padding styles.

## Output

The script outputs the following for each experiment:
- Total Parameters
- FLOPs
- Training Loss
- Test Accuracy

These metrics help compare and identify the most optimal CNN configuration.

## Example Results

| Setting | Parameters | FLOPs | Test Accuracy |
|---------|------------|-------|---------------|
| Setting 1 | X | Y | Z% |
| Setting 2 | X | Y | Z% |
| ...       | ...        | ...   | ...           |

## Future Work

- Extend the dataset to more challenging tasks.
- Experiment with additional convolutional settings such as dilated and depthwise separable convolutions.
- Integrate hyperparameter optimization frameworks.

---

Feel free to contribute to this project by opening issues or submitting pull requests.

# requirements.txt
```
torch
torchvision
matplotlib
```

## Author
This project was created by Seyedalireza Yaghoubi |  David Carranza Navarrete. For any inquiries, please contact syaghoubi@uni-osnabrueck.de | dcarranzanav@uni-osnabrueck.de .
