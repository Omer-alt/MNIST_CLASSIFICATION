# Multi-Class Classification on MNIST
 This repository implements multi-class classification using different approaches:
  -Softmax Regression
  -1-Hidden Layer Neural Network
  -Transfer Learning
  -Others

## Overview
The MNIST dataset is split into even and odd digit sets before classification.
Each model is trained separately on the even and odd subsets to compare performance.


## I. Data processing

Before classification, the MNIST dataset is divided into two subsets:

*Even digits (0, 2, 4, 6, 8)
*Odd digits (1, 3, 5, 7, 9)

## Implemented models
### 1. Softmax regression
Applied to:
#### 1.1 Even digits
![softmax regression for even](/assets/Task-1-even.png)
S
#### 1.2 Odd digits
![softmax regression for odd](/assets/Task-1-odd.png)

### 2. One-hidden layer neural network
Neural Network applied to:
#### 2.1 Even digits
![neural network for even](/assets/Task-2-even.png)

#### 2.2 Odd digits
![neural network for odd](/assets/Task-2-odd.png)


### 3. transfer learning
 Transfer learning is used to transfer knowledge from a model trained on one dataset to another.

#### 3.1 transfer learning from odd pre-trained model to even
![transfer learning for even](/assets/Task-3-even.png)

#### 3.2 transfer learning from even to odd
![transfer learning for odd](/assets/Task-3-Odd.png)

## Performance comparison

| percentage | datatype | Tache1                     | Tache3    |
| :--------  | :------- | :------------------------- |:----------|
|    `10%`   | `even`   |           `95,03`          |  `95,05`  |
|     `10%`  | `odd`    |           `93,17`          |  `97,17`  |
|    `25%`   | `even`   |           `95,43`          |  `95,11`  |
|     `25%`  | `odd`    |           `94,84`          |  `98,01`  |
|    `50%`   | `even`   |           `9,`          |  `9,`  |
|     `50%`  | `odd`    |           `9,`          |  `9,`  |
|    `80%`   | `even`   |           `9,`          |  `9,`  |
|     `80%`  | `odd`    |           `9,`          |  `9,`  |
|    `100%`  | `even`   |           `9,`          |  `9,`  |
|     `100%` | `odd`    |           `9,`          |  `9,`  |

## Optimizations

his project incorporates several optimizations:
- Object-Oriented Programming (OOP) for better modularity and reusability
- Adam Optimizer for efficient gradient descent


## Tech Stack

**Language:** Python

**Package:** Numpy, matplotlib, Pytorch

## Run Locally

Clone the project

```bash
  git clone https://github.com/Omer-alt/MNIST_CLASSIFICATION.git
```

Go to the project directory

```bash
  cd my-project
```

Run the main file

```bash
  main.py
```
## License

[MIT](https://choosealicense.com/licenses/mit/)





















