# SPL-MLL ：Selecting Predictable Landmarks for Multi-Label Learning
A TensorFlow implementation of the paper ["SPL-MLL : Selecting Predictable Landmarks for Multi-Label Learning"](https://arxiv.org/pdf/2008.06883.pdf), ECCV2020.

## Framework
![avatar](https://github.com/yidaiqiushen/SPL-MLL/blob/master/Framework/Framework.png)

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
 
## Training
Enter to the path /model/src, and you can adjust some hyperparameters in `parser.py` and run the code in the following way:  

  `python main.py --load=False`

## Notice
The optimization of this code is without the Alternating Direction Minimization(ADM) strategy. For simplicity, the objective function is passed directly into the Tensorflow optimizer for updating different blocks of variables.
 
## Citation
If you find that SPL-MLL helps your research, please cite our paper.

@inproceedings{li2020landmark,
	title={SPL-MLL: Selecting Predictable Landmarks for
	Multi-Label Learning},
	author={Li, Junbing and Zhang, Changqing and Zhu, Pengfei and Wu, Baoyuan and Hu, Qinghua},
	booktitle={ECCV},
	year={2020},
}

## Questions
For any additional questions, feel free to email lijunbing@tju.edu.cn.

  
 



