# Phase Shift Prediction

[![DOI]()]()

This repository is the official implementation of [Ripple Minimization in Asymmetric Interleaved
DC-DC Converters Using Neural Networks](). 

This study designs a neural network (NN) model predicting phase-shift angle to minimize the total current ripple of the common output capacitor of a multiphase DC-DC converter system. Phase-shift angle prediction is formulated as a regression problem using the system’s operating conditions. The model can be represented as a function parameterized by $\mathcal{W}$ (weights and biases in the network), $f_\mathcal{W}$. The model takes an operating condition $OC = (V_{in1},I_{in1},V_{in2},I_{in2})$ as input, where $V_{in1},V_{in2} \in [15.0V,50.0V]$ and $I_{in1},I_{in2} \in [0.6A,3.0A]$. Then, it predicts the corresponding phase-shift angle $\theta_{pred}=f_{\mathcal{W}}(OC) \in [0\degree,360\degree]$ at the output.

<img src="_images/framework.png" alt="alt text" width="500"/>

 
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
* [AAA BBB](#aaa-bbb)


## Required Python Packages

All the experiments were run in a virtual environment created with pip.

To install requirements:

```console
pip install -r requirements.txt
```

## AAA BBB
