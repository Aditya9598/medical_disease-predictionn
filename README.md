Here’s a detailed **README** for your skin disease image classification project:

---

# Skin Disease Classification Using VGG16

## Project Overview

This project implements a Convolutional Neural Network (CNN) model for multi-class classification of skin diseases using images. The model leverages the pre-trained VGG16 architecture as the base for feature extraction and fine-tunes it to classify skin disease images into several categories. The dataset is split into training and validation sets for model training and evaluation.

## Dataset

The dataset used for this project consists of images classified into multiple skin disease categories. The images are stored in separate folders based on their respective classes.

- **Training Data Path:** `C:\Users\lenovo\Downloads\archive\train`
- **Validation Data Path:** `C:\Users\lenovo\Downloads\archive\test`

Each image is resized to 256x256 pixels to match the input requirements of the VGG16 model.

## Model Architecture

The model is built on top of the **VGG16** architecture pre-trained on the **ImageNet** dataset. Key features of the architecture include:

1. **VGG16 Base:** The pre-trained VGG16 model is used for feature extraction, and its layers are frozen to prevent further training.
2. **Custom Layers:** Several dense layers are added on top of VGG16 for multi-class classification.
   - **Flatten Layer:** Converts the 3D output of the convolutional layers into 1D.
   - **Fully Connected Dense Layers:** Two dense layers with **Batch Normalization** and **Dropout** to prevent overfitting.
   - **Output Layer:** Uses `softmax` activation for multi-class classification.

### Model Summary:

- **Base Model:** VGG16 (without the top fully connected layers)
- **Custom Layers:** Flatten → Dense(512) → BatchNorm → Dropout(0.5) → Dense(256) → BatchNorm → Dropout(0.5) → Dense(num_classes, softmax)
- **Loss Function:** `Sparse Categorical Crossentropy` (for multi-class classification)
- **Optimizer:** Adam (learning rate = 0.001)

## Training

The model is trained on the dataset using the following configurations:

- **Batch Size:** 32
- **Epochs:** 20 (with Early Stopping based on validation loss)
- **Callbacks:**
  - **Early Stopping:** Stops training when validation loss stops improving.
  - **Learning Rate Scheduler:** Gradually decreases the learning rate after 10 epochs to improve convergence.

## Installation and Setup

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Aditya9598/Medical_projectusingcNN.git
   ```

2. **Install Dependencies**:
   Install the necessary Python libraries by running:

   ```bash
   pip install tensorflow matplotlib opencv-python
   ```

3. **Dataset**:
   - Download and place the dataset in the appropriate folders:
     - Training images: `C:\Users\lenovo\Downloads\archive\train`
     - Test images: `C:\Users\lenovo\Downloads\archive\test`

## How to Run the Code

1. **Training the Model**:
   The training script loads the dataset from the specified directories, builds the model, and starts training with early stopping and learning rate scheduling.

   ```python
   model.fit(train_ds, epochs=20, validation_data=validation_ds, callbacks=[early_stopping, lr_scheduler])
   ```

2. **Plot Training/Validation Accuracy and Loss**:
   After training, accuracy and loss plots are generated to visualize the model's performance over the epochs.

3. **Prediction**:
   To predict the disease from a new image, use the `predict_image` function:
   ```python
   predict_image(r'C:\Users\lenovo\Downloads\people_varicella2-lg.jpg')
   ```

## Model Evaluation

- The model is evaluated using validation data and provides accuracy and loss metrics for performance analysis.
- The final model is saved and can be used to predict skin diseases from new images.

## Example Usage

```python
# Predicting the disease for a new image
predict_image(r'C:\path_to_image\example_image.jpg')
```

The model will output the predicted skin disease and display the image with the prediction label.

## Dependencies

- **TensorFlow:** Deep learning framework used to build and train the model.
- **Matplotlib:** For plotting accuracy and loss graphs.
- **OpenCV:** Used for image preprocessing.

## Results

The model can classify skin diseases with reasonable accuracy depending on the dataset and number of classes. You can further fine-tune the model or experiment with additional data augmentation techniques to improve performance.

## Future Work

- Improve the model by unfreezing certain layers of the VGG16 base for fine-tuning.
- Experiment with different architectures or data augmentation techniques to enhance the model’s robustness.
- Expand the dataset for more comprehensive disease detection.
