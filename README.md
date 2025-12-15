# University of Delaware CISC 684 Final Project
This is the repository for a final project for the Introduction to Machine Learning course
at the University of Delaware. The purpose of this project is to compare the accuracy and effectiveness of a
Convolutional Neural Network and a Multi-Class Bayes Classifier. The report for this project can be found [here](https://github.com/masonkulikowski/BayesCNN-Waste/blob/main/A%20Study%20of%20Bayesian%20and%20Convolutional%20Models%20for%20Waste%20Classification.pdf).

## Dataset
I based this project heavily on an existing project and report called [TrashNet](https://github.com/garythung/trashnet), which attempts to solve the same problem I am, but uses slightly different methods. For one, they use an SVM instead of a Bayesian Classifier.

Thankfully, the dataset has been posted on [HuggingFace](https://huggingface.co/datasets/garythung/trashnet), and doesn't need to be added to the repository as HuggingFace allows for direct imports of their datasets. It contains roughly ~5000 images of different classes of recyclables and a trash class. There is a somewhat severe class imbalance in the dataset that made training harder.
- Paper: 23.5%
- Metal: 16.2%
- Glass: 19.8%
- Plastic: 19.1%
- Cardboard: 15.9%
- Trash: 5.4%

## Models

### Convolutional Neural Network

I implemented a custom CNN architecture rather than using a pretrained model like ResNet18. The network consists of 4 convolutional blocks with progressively increasing channel sizes (32 → 64 → 128 → 256). Each block includes batch normalization, ReLU activation, and max pooling to extract hierarchical features. The model uses global average pooling before the fully connected layers to reduce overfitting, followed by dropout for regularization. I trained it using Adam optimizer with a learning rate of 0.001 and implemented early stopping with a patience of 10 epochs to prevent overfitting.

### Bayes Classifier

For comparison, I implemented a Multi-Class Gaussian Naive Bayes classifier with hand-crafted feature extraction. Rather than learning features automatically like the CNN, this approach uses traditional computer vision techniques to extract meaningful features from the images.

The feature extraction pipeline includes:
- **Local Binary Patterns (LBP)** for texture analysis using single-scale LBP with radius 1 and 8 points
- **Color features** from HSV color space histograms
- **Shape features** including aspect ratio and Hu moments
- **Domain-specific features** for detecting specular reflections, metallic surfaces, and glass-like properties

To manage the high dimensionality and remove redundant information, I applied PCA to reduce the features to 15 principal components and removed highly correlated features (threshold 0.85). The classifier uses balanced class priors to account for the class imbalance in the dataset.

# Running the Models

## Prerequisites

[Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) installed on your local machine,
runs project in a virtual environment.

## Commands

Create a virtual environment
```
conda env create -f environment.yml 
```
This will also install all necessary dependicies. 

Activate
```
conda activate bayescnn
```
I was running the Jupyter notebooks through VS Code IDE, but the can also be ran at the command line. This project uses the DirectML library due to the computational restrictions of my machine. Unfortunately I have an AMD GPU so I couldn't parallelize training without using DirectML.
