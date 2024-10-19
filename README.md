
# Currency Image Classification

This project implements a simple image classification model using a Convolutional Neural Network (CNN) to distinguish between images of USD and Euro currency notes. The model is built using OpenCV for image processing and Keras for the neural network.

## Overview

The project involves loading images of currency notes from two classes: USD and Euro. These images are processed, converted to grayscale, and resized to a fixed size. A CNN model is then trained to classify the images into one of the two categories.

## Dataset

The images used for training should be organized in the following structure:

```
TrainingSet/
    ├── USD/
    │   ├── image1.jpg
    │   ├── image2.png
    │   └── ...
    └── Euro/
        ├── image1.jpg
        ├── image2.png
        └── ...
```

Each folder should contain the respective currency images in either `.jpg` or `.png` format.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Olamide643/Paper-Currency-Recognition.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` should include:
   - `numpy`
   - `opencv-python`
   - `pandas`
   - `keras`
   - `tensorflow`

3. Ensure you have a dataset in the correct format.

## Usage

Run the script to train the model:

```bash
python train.py
```

The script will:
- Load the images from the `TrainingSet` directory.
- Preprocess the images (convert to grayscale, resize to 64x64 pixels).
- Shuffle and split the data into features and labels.
- Train a CNN to classify the images as either USD (label `0`) or Euro (label `1`).
- The model summary is displayed after compilation.

## Model Architecture

The CNN model has the following architecture:
- Input layer: 64x64 grayscale images.
- 2 Convolutional layers with 32 filters each, followed by a max-pooling layer.
- A fully connected Dense layer with 64 units and ReLU activation.
- A dropout layer to reduce overfitting.
- An output Dense layer with sigmoid activation for binary classification.

## Contributing

If you'd like to contribute to this project, please feel free to submit a pull request or open an issue.

