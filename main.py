# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from neural_network import SoftmaxRegressionNN
from softmax_regression import SoftmaxRegression
from transfert_learning import TransfertLearning

# Set random seed for reproducibility
torch.manual_seed(42)

# Define device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import utility classes



"""_summary_: Use of the design pattern, in particular the Singleton pattern, for better management of MNIST data
    Returns:
        _type_: _description_
"""

class DataHandler:
  _instance = None

  def __init__(self):
    pass

  def __new__(cls):
    if cls._instance is None:
      print('Creating the object')
      cls._instance = super(DataHandler, cls).__new__(cls)
      cls._instance.mnist_train = datasets.MNIST(root="data", train=True, transform=transforms.ToTensor(), download=True)
      cls._instance.mnist_test = datasets.MNIST(root="data", train=False, transform=transforms.ToTensor(), download=True)
      cls._instance.mnist_train_even = None
      cls._instance.mnist_train_odd = None
      cls._instance.filter_parity()

    return cls._instance

  def filter_parity(self):
    filter_even = lambda label: label % 2 == 0
    self.mnist_train_even = [(img, label) for img, label in self.mnist_train if filter_even(label)]
    self.mnist_train_odd = [(img, label) for img, label in self.mnist_train if (~filter_even(label))]

# Define the transformation to apply to the data
torch.manual_seed(42)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train = True, transform = transform, download = True)
test_dataset = datasets.MNIST(root='./data', train = False, transform = transform)

# Function to filter even and odd numbers
def filter_even_odd(dataset, even=True):
    filtered_data = []
    if even:
      new_labels = {0: 0 , 2: 1, 4: 2 , 6: 3 , 8: 4}
    else:
      new_labels = {1: 0 , 3: 1 , 5: 2 , 7: 3, 9: 4}
    for image, label in dataset:
        if (label % 2 == 0) == even:
            new_label = new_labels[label]
            filtered_data.append((image, new_label))
    return filtered_data

# Filter even and odd numbers from the dataset
train_even_dataset = filter_even_odd(train_dataset, even = True)
train_odd_dataset = filter_even_odd(train_dataset, even = False)
test_even_dataset = filter_even_odd(test_dataset, even = True)
test_odd_dataset = filter_even_odd(test_dataset, even = False)

# Define the batch size for training
batch_size_even = len(train_even_dataset)
batch_size_odd = len(train_odd_dataset)
# batch_size = 6000

# Create data loaders for even and odd datasets
train_even_loader = torch.utils.data.DataLoader(train_even_dataset, batch_size = batch_size_even, shuffle = True)
train_odd_loader = torch.utils.data.DataLoader(test_odd_dataset, batch_size = batch_size_odd, shuffle = True)
test_even_loader = torch.utils.data.DataLoader(test_even_dataset, batch_size = batch_size_even, shuffle = False)
test_odd_loader = torch.utils.data.DataLoader(train_odd_dataset, batch_size = batch_size_odd, shuffle = False)


def main():
    
    
    """
    ____Prepared data for train models 
    - Softmax regression
    - 1-hidden layer neural network for softmax classificaation
    - Transfert learning between odd and even pre-trained models
    """
    menu = ["- Softmax regression on even MNIST data", "- Softmax regression on odd MNIST data",  "- 1-hidden layer neural network for softmax classificaation on even MNIST data", "- 1-hidden layer neural network for softmax classificaation on odd MNIST data",  "- Transfert learning from odd model and even model ", "- Transfert learning from even model and odd model " ]
    
    print("_________ Hello ___________", end="\n")
    [print(i, menu[i], end="\n") for i in range(len(menu))]
    
    choice = int(input("choose what you want to run: "))
    while choice > len(menu):
        choice = int(input(f"choose what you want to run from 0 to: {len(menu) - 1}"))
    
    
    # model parameters ( common parameters )
    
    input_size = 28 * 28
    hidden_size = 500
    num_classes = 5
    num_epochs= 500
    num_epochs_n = 100
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    
    # create pretraining models
    pre_train_even = None
    pre_train_odd = None
        
    
    match choice:
        case 0:
            print("-********** Softmax regression on even MNIST data **********")
            
            # even  models
            even_model = SoftmaxRegression(input_size, num_classes).to(device)

            # loss function and optimizer
            even_optimizer = optim.Adam(even_model.parameters(), lr=0.001)
            

            # Train the even model
            print("Training even model...")
            even_model.train(even_model, train_even_loader, test_even_loader,criterion, even_optimizer, num_epochs)

            # Plot the train and the test loss curves for evens
            even_model.plot_function()
            
            return 
        
        case 1 :
            print("-********** Softmax regression on odd MNIST data **********")
            
            # even  models
            odd_model = SoftmaxRegression(input_size, num_classes).to(device)
            odd_optimizer = optim.Adam(odd_model.parameters(), lr=0.001)
            
            # Train the odd model
            print("Training odd model.............")
            odd_model.train(odd_model, train_odd_loader, test_odd_loader, criterion, odd_optimizer, num_epochs)

            # Plot the train and the test loss curves for evens
            odd_model.plot_function()
            
            return
            
            
        case 2 :
            print("-********** 1-hidden layer neural network for softmax classificaation on even MNIST data **********")
            
            # Initialize the model
            even_model = SoftmaxRegressionNN(input_size, hidden_size, num_classes)
            
            # Optimizer for even and odd
            optimizer_even = optim.Adam(even_model.parameters(), lr=learning_rate)

            # Plot even losses
            even_model.training_process(even_model, train_even_loader, test_even_loader, criterion, optimizer_even, num_epochs_n, even=True)
            
            if even_model.is_train:
                pre_train_even = even_model
                
            even_model.plot_function()
            
            return 
        
        case 3 :
            print("-********** 1-hidden layer neural network for softmax classificaation on odd MNIST data **********")
            
            # Initialize the model
            odd_model = SoftmaxRegressionNN(input_size, hidden_size, num_classes)
            
            # Optimizer for even and odd
            optimizer_odd = optim.Adam(odd_model.parameters(), lr=learning_rate)
            
            
            # Plot odd losses
            odd_model.training_process(odd_model, train_odd_loader, test_odd_loader, criterion, optimizer_odd, num_epochs_n, even=False)
            
            if odd_model.is_train:
                pre_train_odd = odd_model
                
            odd_model.plot_function()
            
            return 
        
        case 4 :
            print("-********** Transfert learning from odd model and even model **********")
            
            # Be sure that we have pre-trained model others wise will train them
            if pre_train_even == None :
                #  Train the even model before
                print("-********** We need 1-hidden layer neural network for softmax classificaation on even MNIST data **********")
                pre_train_even = SoftmaxRegressionNN(input_size, hidden_size, num_classes)
                
                # Optimizer for even and odd
                optimizer_even = optim.Adam(pre_train_even.parameters(), lr=learning_rate)
                
                pre_train_even.training_process(pre_train_even, train_even_loader, test_even_loader, criterion, optimizer_even, num_epochs_n, even=True)
                
                
            if pre_train_odd == None :
                #  Train the odd model before
                print("-********** We need 1-hidden layer neural network for softmax classificaation on odd MNIST data **********")
                pre_train_odd = SoftmaxRegressionNN(input_size, hidden_size, num_classes)
                
                # Optimizer for even and odd
                optimizer_odd = optim.Adam(pre_train_odd.parameters(), lr=learning_rate)
                
                pre_train_odd.training_process(pre_train_odd, train_even_loader, test_even_loader, criterion, optimizer_even, num_epochs_n, even=False)
                
            
            # Models that must learn from existing models
            even_model1 = SoftmaxRegressionNN(input_size, hidden_size, num_classes)
            odd_model1 = SoftmaxRegressionNN(input_size, hidden_size, num_classes)

            # Call it for odd data (pass and even model)
            transfert_from_even_to_odd = TransfertLearning(pre_train_even, even_model1)

            transfert_from_even_to_odd.train_frozen_model(even_model1, train_odd_loader, test_odd_loader, criterion, num_epochs, even= False)
            transfert_from_even_to_odd.model.plot_function()

            # Call it for even data (pass and odd model)
            transfert_from_odd_to_even = TransfertLearning(pre_train_odd, odd_model1) 

            transfert_from_odd_to_even.train_frozen_model(odd_model1, train_even_loader, test_even_loader,criterion, num_epochs_n, even= True)
            transfert_from_odd_to_even.model.plot_function()
            
            return 
        
        case 5 :
            print("-********** Transfert learning from even model and odd model **********")
            
            
            
            return 
        case _ :
            return

    
if __name__ == "__main__":
    main()
    
    





