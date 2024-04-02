import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from neural_network import SoftmaxRegressionNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransfertLearning:
    def __init__(self, pre_trained_model, model ):
        self.pre_trained_model = pre_trained_model
        self.model = model
        
    # transfer of parameters
    def parameters_transfer(self):
        try:
            # save the pre-trained model parameters
            torch.save(self.pre_trained_model.state_dict(), 'pre_trained_model.pth')
            self.model.load_state_dict(torch.load('pre_trained_model.pth'))
            self.model.eval()
            
            ## freeze the parameters:
            for param in self.model.layer1.parameters():
                param.requires_grad = False
                
            
        except Exception as e:
            print("Something when wrong when making the transfer of parameters")
        
               
    def train_frozen_model(self, model, loader, test_loader,criterion, learning_rate, num_epochs=300, even= True):
        
        self.parameters_transfer()
        
        # Adam optimizer
        optimizer = optim.Adam(self.pre_trained_model.parameters(), lr=learning_rate)
        
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

            # # Test the model
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():

                    for images, labels in test_loader :
                        outputs = model(images.view(-1, 784))
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)

                        #outputs = model(images)
                        t_loss = criterion(outputs, labels)
                        model.test_losses.append(t_loss.item())

                        correct += (predicted == labels).sum().item()

            if epoch  % 10 == 0:
                print(f'Epoch [{epoch}/{num_epochs}], Test Accuracy: {(correct/total)*100:.2f}%')
            # print('Accuracy :{:.2f}%'.format((correct/ total)*100))




