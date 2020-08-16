# SPL-MLL ：SPL-MLL: Selecting Predictable Landmarks for Multi-Label Learning
A TensorFlow implementation of the paper "SPL-MLL ：SPL-MLL: Selecting Predictable Landmarks for Multi-Label Learning", ECCV2020.

## Framework

## Requirements
- TensorFlow 1.3+  
- Python 3.5  
- sklearn  
- numpy  
- scipy
- xlwt

## Introduction
Although significant progress achieved, multi-label classification is still challenging due to the complexity of correlations among different labels. Furthermore, modeling the relationships between input and some (dull) classes further increases the difficulty of accurately predicting all possible labels. In this work, we propose to select a small subset of labels as landmarks which are easy to predict according to input (predictable) and can well recover the other possible labels (representative).
Different from existing methods which separate the landmark selection and landmark prediction in the 2-step manner, the proposed algorithm, termed Selecting Predictable Landmarks for Multi-Label Learning (SPL-MLL), jointly conducts landmark selection, landmark prediction, and label recovery in a unified framework, to ensure both the representativeness and predictableness for selected landmarks. We employ the Alternating Direction Method (ADM) to solve our problem. Empirical studies on real-world datasets show that our method achieves superior classification performance over other state-of-the-art methods.

## Example Experiments
This repository contains a subset of the experiments mentioned in the paper.

## Testing
Enter to the path /model/src, and you can simply run the code in the following way:  
`python main.py --load=False`



