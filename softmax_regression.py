import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#softmax regression model
class SoftmaxRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.train_losses = []
        self.test_losses = []


    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out

    def plot_function(self):
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
    #training function
    def train(self, model, loader, test_loader,criterion, optimizer, num_epochs):
        # total_step = len(loader)
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(loader):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images).to(device)
                loss = criterion(outputs, labels)
                train_loss = loss.item()
                model.train_losses.append(train_loss)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch) % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

            with torch.no_grad():
                total = 0
                correct = 0
                for images, labels in test_loader:
                    outputs = model(images)
                    test_loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()


                model.test_losses.append(test_loss.item())
            if (epoch ) % 10 == 0:
                print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {(correct/total)*100:.2f}%')
                
                

