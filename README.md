# Hand-Drawn Shapes Classifier

## Overview
This project aims to train a Convolutional Neural Network (CNN) to classify hand-drawn shapes. The dataset used in this project can be found in the `dataset` folder.

## Dataset
The dataset consists of images of different geometric shapes, including circles, kites, parallelograms, squares, trapezoids, and triangles. The dataset is divided into training, validation and testing sets, located in the `dataset/train`, `dataset/val` and `dataset/test` directories, respectively. <br>
An additional small hand-crafted dataset was created which is located in `dataset/personal_drawing`.

## Model Training
The CNN model architecture is defined in `CNNmodel.py`. The model is trained using mini-batches, with the training loss and accuracy displayed during training. The trained model achieves an accuracy of ~ 67% against the hand-drawn shape dataset. <br>
The model was designed to be run on a CUDA-capable device in order to reduce training times. <br> 
See more about installing CUDA on your device at https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html

## Running the Code
To run the code, follow these steps:
1. Ensure you have Python 3 and the required dependencies installed.
2. Clone this repository to your local machine.
3. Open a terminal or command prompt and navigate to the project directory.
4. Run the following command to execute the code:

## Output
The output of the code includes the training progress, such as the loss and accuracy at each mini-batch, as well as the validation loss and accuracy after each epoch. Additionally, a plot of the training and validation loss and accuracy is generated.<br> 
Furthermore, a confusion matrix of the model evaluated against the testing data is generated.

## Future Improvements
The model’s achieved performance could be attributed to the following factors:
1.	Dataset mismatch: The hand-drawn shapes dataset and the dataset on which the model was trained have different characteristics, styles, and variation. 
2.	Model Architecture limitations: The chosen model may have been too simple and was not suitable for the complexities and variations present in the hand-drawn shapes dataset.
3.	Limited training dataset: The dataset on which the model was trained was small which could have led to its poor performance. <br>

From the above-identified factors, the following steps could be taken to improve the model’s performance:
1.	Create a full hand-drawn shapes dataset that closely resembles the target domain/use-case and retrain the model on this dataset.
2.	Increase the complexity of the model but adding additional full connected layers. Having more than 1 fully connected layer would allow for more non-linearity to be present in the model, which would vastly improve the model’s performance. Additional convolutional layers could also be experimented with.
3.	Increase the size of the dataset on which the model was trained. 
