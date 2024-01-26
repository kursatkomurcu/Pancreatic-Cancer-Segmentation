# Pancreatic-Cancer-Segmentation

## Introduction

Pancreatic cancer is a devastating disease with a high mortality rate, making early detection and
accurate segmentation crucial for effective treatment and prognosis. Computed Tomography (CT)
scans have emerged as a valuable tool for the diagnosis and management of pancreatic cancer. In
recent years, the field of medical image analysis has witnessed significant advancements,
particularly in the domain of deep learning techniques. The utilization of architectural neural
network designs, such as the UNet model, and the fine-tuning of hyperparameters have played a
pivotal role in enhancing the accuracy and efficiency of pancreatic cancer segmentation from CT
scans. This assignment seeks to investigate the impact of various architectural neural network
designs and hyperparameters on the segmentation results of pancreatic cancer, contributing to the
ongoing efforts to improve the diagnostic outcomes for this aggressive disease. To date, several
studies have leveraged deep learning techniques in medical imaging, demonstrating their potential
to achieve remarkable segmentation results, thus motivating this study's pursuit of refining these
methodologies [1, 2].

## Data And Preprocessing

In the context of this assignment, the Medical Decathlon Task07_Pancreas dataset [3]3 was
employed, comprising a total of 281 training files, 281 corresponding label files, and 139 test files.
Notably, these files are formatted in NIfTI (Neuroimaging Informatics Technology Initiative), a
standard data format that has gained widespread adoption in the field of neuroimaging, particularly
for brain imaging, such as MRI scans. NIfTI's utility lies in its capacity to encapsulate both image
data and crucial metadata pertaining to the scan within a single file. This integrated format greatly
simplifies data handling and facilitates seamless data exchange across various software platforms
and research initiatives, making it an invaluable asset within the neuroscience community.

To optimize data preparation and alleviate computational demands, the initial step involved the
development of a Python script that efficiently decompressed all NIfTI files. Subsequently, a data
preprocessing step was executed, where the width and height of each image were scaled to 256
pixels. This dimension reduction was motivated by a desire to manage computational resources
judiciously, thereby expediting the experimental workflow.

Furthermore, in light of limited available RAM capacity, a judicious subset of the dataset consisting
of 105 training samples, 105 corresponding labels, and 42 test data samples was selected for
conversion into Numpy file format in initial result report. This selective sampling approach allowed
for the expeditious generation of Numpy files, effectively streamlining the subsequent training
process. These Numpy files are three-dimensional, with each channel containing distinct images
and labels, laying the foundation for subsequent investigations into pancreas cancer segmentation. 

For a more comprehensive analysis, we utilized the entire dataset, encompassing 281 training
samples and 139 test samples. This broader utilization aims to enhance the depth and accuracy of
our findings in the realm of pancreas cancer segmentation.

## Training And Results

Throughout the training phase, a Google Colab V100 GPU was employed in conjunction with the
U-Net architecture to facilitate pancreas cancer segmentation. U-Net, an innovative convolutional
neural network-derived framework, has consistently exhibited superior efficacy in pixel-based
image segmentation tasks compared to conventional models. Initially designed for biomedical
image analysis, its unique architecture has found adaptation across various segmentation
applications.

The U-Net model's feature extraction employs convolutional layers in its first half, succeeded by
dimensionality reduction through pooling layers. Notably, this reduction is inverted in the latter
half, resulting in dimensionality expansion, strategically enhancing output resolution. Additionally,
the model incorporates skip connections that interconnect downsampled output with high-resolution
features, contributing significantly to precise localization. [1]

In my previous experiments, I explored the performance of the U-Net model, as well as variations
such as a simplified U-Net, FCN (Fully Convolutional Network), and VNet. U-Net, designed for
biomedical imaging, excels in capturing intricate features. FCN, a pioneering architecture for
semantic segmentation, operates on the entire image without the need for fully connected layers.
VNet, tailored for volumetric medical image segmentation, extends the principles of U-Net to 3D
data. In this updated analysis, I aim to compare these models to identify the most effective approach
for pancreas cancer segmentation. [4, 5]

![unet](https://github.com/kursatkomurcu/Pancreatic-Cancer-Segmentation/blob/main/unet.png)

![fcn](https://github.com/kursatkomurcu/Pancreatic-Cancer-Segmentation/blob/main/fcn.png)

![vnet](https://github.com/kursatkomurcu/Pancreatic-Cancer-Segmentation/blob/main/vnet.png)


The Dice coefficient, renowned for its effectiveness in segmentation tasks, was utilized
to quantitatively assess the agreement between the predicted segmentation and the ground truth.
The Dice coefficient, a pivotal metric in our assessment, is calculated by measuring the degree of
pixel-wise overlap between the predicted segmentation and the corresponding ground truth.

Specifically, it is computed as twice the area of overlap between these two images, divided by the
total number of pixels collectively encompassed by both images. The utilization of the Dice
coefficient facilitates a comprehensive evaluation of the segmentation model's performance by
quantifying the degree of correspondence between the predicted outcomes and the ground truth,
thereby offering insights into the model's accuracy and effectiveness. After initial results, the
smooth value was decreased to reach more realistic results.

For the optimization of the neural network, the Adam optimizer was employed to fine-tune the
model's parameters and enhance its convergence during training. Additionally, in the convolutional
layers, the Rectified Linear Unit (ReLU) activation function was chosen to introduce non-linearity,
aiding in feature extraction, while the sigmoid activation function was adopted in the output layer to
ensure that the model's predictions fell within the appropriate range. Furthermore, for the initial
training phase, a predetermined number of epochs were set at 30, and the data was divided into
training and validation sets with a split parameter of 0.3, signifying that 30% of the data was
reserved for testing purposes.Remarkably, the training process was accomplished within a relatively
short span of approximately 80 minutes, culminating in a vital step toward our research objective.

![dice](https://github.com/kursatkomurcu/Pancreatic-Cancer-Segmentation/blob/main/dice.png)

In order to optimize performance and mitigate overfitting risks, several enhancements were
incorporated into the model configurations. Dropout layers were strategically introduced to each
model, providing a regularization mechanism. Additionally, L2 regularization was applied to the
convolutional layers to encourage a more robust learning process. The validation data ratio was
increased to 0.3 to ensure a more representative assessment of model generalization. To fine-tune
the training dynamics, the learning rate was adjusted to 2e-4. The entire dataset was utilized for
training to harness the full potential of the available information. A comprehensive investigation
was conducted using four distinct models: U-Net, a simplified U-Net, Fully Convolutional Network(FCN), and 2-D VNet. These models were selected to explore diverse architectural paradigms and
their effectiveness in pancreas cancer segmentation.

## Results

![train_result](https://github.com/kursatkomurcu/Pancreatic-Cancer-Segmentation/blob/main/train_result.png)

![test_result](https://github.com/kursatkomurcu/Pancreatic-Cancer-Segmentation/blob/main/test_result.png)

## Codes

**unzip.py:** Unzip all .nii.gz files

**dataset.py:** Convert all .nii files to a numpy file as train data, mask data and test data

**train_and_test.ipynb:** Train segmentation model and plot the results

## Referances:

1- Olaf Ronneberger, Philipp Fischer, and Thomas Brox, U-Net Convolutional Networks for
Biomedical Image Segmentation. https://arxiv.org/pdf/1505.04597.pdf

2- Litjens, G., Kooi, T., Bejnordi, B. E., Setio, A. A. A., Ciompi, F., Ghafoorian, M., ... & Sanchez,
C. I. (2017). A survey on deep learning in medical image analysis. Medical image analysis
https://arxiv.org/pdf/1702.05747.pdf

3- https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2

4- Jonathan Long, Evan Shelhamer, Trevor Darrell, Fully Convolutional Networks for Semantic
Segmentation. https://arxiv.org/pdf/1411.4038.pdf

5- Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi, V-Net: Fully Convolutional Neural Networks for
Volumetric Medical Image Segmentation. https://arxiv.org/pdf/1606.04797.pdf

6- https://www.kaggle.com/code/abhinavsp0730/semantic-segmentation-by-implementing-fcn

7- https://github.com/FENGShuanglang/2D-Vnet-Keras/tree/master
