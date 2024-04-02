import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SoftmaxRegressionNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SoftmaxRegressionNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)
        self.train_losses = []
        self.test_losses = []
        self.is_train = False

    def forward(self, x):
        z1 = self.layer1(x)
        A1 = self.relu(z1)
        Z2 = self.layer2(A1)
        return Z2

    def plot_function(self):
      plt.plot(self.train_losses, label='Training Loss')
      plt.plot(self.test_losses, label='Test Loss')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.legend()
      plt.show()
      
    def training_process(self, model, loader, test_loader ,criterion, optimizer, num_epochs, even=True):

        if even:
            print("Training even model...")
        else:
            print("Training odd model...")

        for epoch in range(num_epochs):
            total_step = len(loader)
            for _, (images, labels) in enumerate(loader):
                images = images.view(-1, 784).to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                model.train_losses.append(loss.item())


                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch  % 10 == 0:
                print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

            # Change the state of my model
            self.is_train = True
            
            # Test the model
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:
                    outputs = model(images.view(-1, 784))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)

                    #outputs = model(images)
                    t_loss = criterion(outputs, labels)
                    model.test_losses.append(t_loss.item())

                    correct += (predicted == labels).sum().item()

            if epoch  % 10 == 0:
                print(f'Epoch [{epoch}/{num_epochs}], Test loss:{t_loss}, Test Accuracy: {(correct/total)*100:.2f}%')












