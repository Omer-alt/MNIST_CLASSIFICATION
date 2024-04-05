# implementation of multi-class classification by training the soft-max regression model and neural network on MNIST dataset.


This repository presents the implementation of multi-class classification in different ways: softmax regression, 1-hidden layer neural network and transfer learning...

## I. Data processing

MNIST data were separated into even and odd sets before classification

### 1. Softmax regression
#### 1.1 Softmax regression for even
![softmax regression for even](/assets/Task-1-even.png)
S
#### 1.2 Softmax regression for odd
![softmax regression for odd](/assets/Task-1-odd.png)

## 2. One-hidden layer neural network

### 2.1 One-hidden layer neural network for even
![neural network for even](/assets/Task-2-even.png)

### 2.2 One-hidden layer neural network for odd
![neural network for odd](/assets/Task-2-odd.png)


## 3. transfer learning
 
### 3.1 transfer learning from odd pre-trained model to even
![transfer learning for even](/assets/Task-3-even.png)

### 3.2 transfer learning from even to odd
![transfer learning for odd](/assets/Task-3-Odd.png)

#### comparison summary in a table

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

What about optimizations in this code ? You can notice the usage of
-  OOP paradigm
- Adam Gradient descent


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





















