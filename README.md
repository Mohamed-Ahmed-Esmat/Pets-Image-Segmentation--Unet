# U-Net Implementation for Image Segmentation using Oxford-IIIT Pet Dataset

This repository contains an implementation of the **U-Net** architecture for image segmentation tasks using the **Oxford-IIIT Pet Dataset**. U-Net is commonly used in image segmentation applications, such as medical imaging and object boundary detection. The dataset contains images of pets with corresponding masks that include class labels and segmentation boundaries.

## Features

- **Data Preprocessing**: Load and preprocess the Oxford-IIIT Pet dataset, which includes images and segmentation masks.
- **Model Architecture**: Build and train a U-Net model using TensorFlow and Keras with encoder-decoder architecture and skip connections.
- **Training and Evaluation**: Train the model and evaluate its performance on a validation set. Visualize predictions against ground truth masks.

## Structure of the Notebook

1. **Data Loading**:
   - Functions to load images and masks from the Oxford-IIIT Pet dataset.
   - Preprocessing includes resizing images and masks to a uniform shape, normalizing pixel values, and splitting data into training and test sets.

2. **Model Architecture**:
   - The U-Net model consists of an encoder that progressively reduces the spatial dimensions and a decoder that reconstructs the image using upsampling.
   - Skip connections ensure that detailed spatial information is retained during the reconstruction.

3. **Model Training**:
   - The model is trained using binary cross-entropy loss, a common loss function for segmentation tasks with binary masks.
   - After training, model performance is evaluated using accuracy metrics and visual comparison between predictions and ground truth masks.
   - The model was trained for 20 epochs

4. **Visualization**:
   - The notebook includes code to visualize the segmentation results, allowing comparison of the model's predictions with the true labels.

## Oxford-IIIT Pet Dataset

The dataset consists of 37 categories of pets with pixel-level masks. Each image comes with a corresponding mask that annotates the object class and the segmentation boundary.

Download the dataset from the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).

## Requirements

- TensorFlow
- Keras
- NumPy
- Matplotlib
- PIL
- ImageIO

To install the required libraries, run:
```bash
pip install tensorflow numpy matplotlib pillow imageio
```
## Results
The loss and accuracy graphs:
![image](https://github.com/user-attachments/assets/7910e463-a2ce-42e6-9cad-eecb03cf7a16)

Here's a sample from the output:
![image](https://github.com/user-attachments/assets/228463f2-8fc5-4d58-8c1a-a6fe9895bb13)

## Acknowledgements
This implementation uses the U-Net architecture introduced in the paper [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597) and leverages the Oxford-IIIT Pet Dataset for segmentation tasks.
