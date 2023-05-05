# Phase Shift Prediction

[![DOI]()]()

This repository is the official implementation of [Ripple Minimization in Asymmetric Interleaved
DC-DC Converters Using Neural Networks](). 

This study designs a neural network (NN) model predicting phase-shift angle to minimize the total current ripple of the common output capacitor of a multiphase DC-DC converter system. Phase-shift angle prediction is formulated as a regression problem using the system’s operating conditions. 

<img src="_images/two_boost.png" alt="alt text" width="300"/>

Folder structure:

```console
Phase Shift Prediction
├── _images
├── data
├── phase_shift_prediction
├── README.md
└── requirements.txt
```

We will explain the following steps one-by-one:

* [Required Python Packages](#required-python-packages)
* [Machine Learning Dataset](#machine-learning-dataset)
* [Phase Shift Prediction](#phase-shift-prediction)


## Required Python Packages

All the experiments were run in a virtual environment created with pip.

To install requirements:

```console
pip install -r requirements.txt
```

## Machine Learning Dataset

Let $\mathcal{D}$ be a machine learning dataset such that for each $(OC,\theta_{opt}) \in \mathcal{D}$, $OC=(V_{in1},~I_{in1},~V_{in2},~I_{in2})$ is an operating condition of the converter system and $\theta_{opt} \in [0\degree, 360\degree]$ is the corresponding optimal phase-shift angle to minimize the total current ripple of the common output capacitor. The input voltages and currents are $V_{in1}, V_{in2} \in \\{15.0V, 16.0V, \cdots, 50.0V\\}$ and $I_{in1}, I_{in2} \in \\{0.6A, 0.8A, \cdots, 2.0A\\}$, respectively. Hence, there are 219024 data points in the machine learning dataset, i.e., $|\mathcal{D}|=219024$. Then, the dataset is randomly segregated into three disjoint sets: training ($\mathcal{D}_{train}$), validation ($\mathcal{D}_{valid}$), and test ($\mathcal{D}_{test}$). There are 146016, 24336, and 48672 data points in training, validation and test sets, respectively. Please note that min-max normalization is applied on the raw data before feeding to the network.

The "data" folder stores the machine learning dataset. 

```console
data
├── data_train_normalized.txt
├── data_train_raw.txt
├── data_valid_normalized.txt
├── data_valid_raw.txt
├── data_test_normalized.txt
└── data_test_raw.txt
```

A few example data points from the training set are shown below.

| $V_{in1}$ | $I_{in1}$ | $V_{in2}$ | $I_{in2}$ | $\theta_{opt}$|
| --- | --- | --- | --- | --- |
| 15.00 | 0.60 | 15.00 | 1.40 | 90.00 |
| 15.00 | 0.60 | 15.00 | 1.60 | 90.00 |
| 15.00 | 0.60 | 15.00 | 1.80 | 90.00 |
| 15.00 | 0.60 | 15.00 | 2.00 | 90.00 |


## Phase Shift Prediction

The model can be represented as a function parameterized by $\mathcal{W}$ (weights and biases in the network), $f_\mathcal{W}$. The model takes an operating condition $OC = (V_{in1},I_{in1},V_{in2},I_{in2})$ as input, where $V_{in1},V_{in2} \in [15.0V,50.0V]$ and $I_{in1},I_{in2} \in [0.6A,3.0A]$. Then, it predicts the corresponding phase-shift angle $\theta_{pred}=f_{\mathcal{W}}(OC) \in [0\degree,360\degree]$ at the output.

<img src="_images/framework.png" alt="alt text" width="500"/>




