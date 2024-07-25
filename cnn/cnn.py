import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Dict, Any
import torch.utils.data
from torchvision import datasets, transforms
from pathlib import Path

def get_transforms(grayscale: bool = False):
    if grayscale:
        train_transforms = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor()
            ]
        )
    else:
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )
    test_transforms = train_transforms
    return train_transforms, test_transforms

class RegressionDataset(torch.utils.data.Dataset):
  def __init__(self, images, targets, transform=None):
    self.images = np.load(images)
    self.targets = np.load(targets)
    self.transform = transform

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    image = self.images[idx]
    #print(image)
    target = self.targets[idx]

    if self.transform:
      image = self.transform(image)  # Apply transformations if provided

    image = torch.from_numpy(np.array(image)).float()  # Convert to PyTorch tensor (float32)
    target = torch.from_numpy(np.array(target)).float()  # Convert to PyTorch tensor (float32)

    return image, target

class RegressionTaskData:
    def __init__(self, grayscale: bool = False) -> None:
        self.grayscale = grayscale
        self.image_folder_path: Path = Path("dataset/")
        self.train_transforms, self.test_transforms = get_transforms(grayscale)
        self.trainloader = self.make_trainloader()
        self.testloader = self.make_testloader()

        
    def make_trainloader(self):
        train_dataset = RegressionDataset(
            self.image_folder_path / 'train_images.npy',  # Assuming your images are in a single file
            self.image_folder_path / 'train_targets.npy',
            transform=self.train_transforms
        )
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle = True)
        return trainloader
    
    def make_testloader(self):
        test_dataset = RegressionDataset(
            self.image_folder_path / 'test_images.npy',
            self.image_folder_path / 'test_targets.npy',
            transform=self.test_transforms
        )
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
        return testloader
    def visualize_image(self):
        """
        This function visualizes a single image from the train set
        """
        images, targets = next(iter(self.trainloader))
        print(targets[0].shape)
        print(images[0].shape)
        if self.grayscale:
            plt.imshow(images[0][0, :, :], cmap='gray')
        else:
            plt.imshow(images[0].permute(1, 2, 0))
        plt.show()


class CNNRegression(nn.Module):
    
    def __init__(self, image_size: Tuple[int, int, int] = (4, 135, 135)):
        super(CNNRegression, self).__init__()
        self.image_size = image_size
        self.conv1 = nn.Conv2d(in_channels=self.image_size[0], out_channels=4, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1_in = int(16*(image_size[1]//16)*(image_size[2]//8))
        self.fc1 = nn.Linear(in_features=self.fc1_in, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)

        
    def forward(self, x):
       
        x = self.conv1(x)
        # print('Size of tensor after each layer')
        #print(f'conv1 {x.size()}')
        x = nn.functional.relu(x)
        #print(f'relu1 {x.size()}')
        x = self.pool1(x)
        #print(f'pool1 {x.size()}')
        x = self.conv2(x)
        #print(f'conv2 {x.size()}')
        x = nn.functional.relu(x)
        #print(f'relu2 {x.size()}')
        x = self.pool2(x)
        #print(f'pool2 {x.size()}')
        x = self.conv3(x)
        #print(f'conv3 {x.size()}')
        x = nn.functional.relu(x)
        #print(f'relu3 {x.size()}')
        x = self.pool3(x)
        #print(f'pool3 {x.size()}')
        x = self.conv4(x)
        #print(f'conv4 {x.size()}')
        x = nn.functional.relu(x)
        #print(f'relu4 {x.size()}')
        x = self.pool4(x)
        #print(f'pool4 {x.size()}')
        x = x.view(-1, self.fc1_in)
        # print(f'view1 {x.size()}')
        x = self.fc1(x)
        # print(f'fc1 {x.size()}')
        x = nn.functional.relu(x)
        # print(f'relu2 {x.size()}')
        x = self.fc2(x)
        # print(f'fc2 {x.size()}')
        return x


history = []
def train_network(device, n_epochs: int = 10, image_size: Tuple[int, int, int] = (4, 135, 135)):
    """
    This trains the network for a set number of epochs.
    """
    if image_size[0] == 1:
        grayscale = True
    else:
        grayscale = False
    assert image_size[1] == image_size[2], 'Image size must be square'
    
    regression_task = RegressionTaskData(grayscale=grayscale)

    # Define the model, loss function, and optimizer
    model = CNNRegression(image_size=image_size)
    model.to(device)
    print(model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    #writer = SummaryWriter()
    starttt = time.time()
    best_loss = np.inf
    best_weights = None
    for epoch in range(n_epochs):
        start = time.time()
        for i, (inputs, targets) in enumerate(regression_task.trainloader):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            #writer.add_scalar('Train Loss', loss.item(), i)

            # Print training statistics
            if i == len(regression_task.trainloader)//2:
                print(f'Epoch [{epoch + 1}/{n_epochs}], Step [{i + 1}/{len(regression_task.trainloader)}], Loss: {loss.item():.7f}')
        _, _, _, mean_distance_loss = evaluate_network(model, device, image_size = image_size)
        if mean_distance_loss < best_loss:
            best_loss = mean_distance_loss
            best_weights = copy.deepcopy(model.state_dict())
        duration = time.time()-start
        print(f'Epoch [{epoch+1}/{n_epochs}] finished in {duration:.5f} seconds, mean distance loss: {mean_distance_loss:.7f} meters\n-----')
        history.append(mean_distance_loss)
    #writer.close()
    total_time = time.time() - starttt
    print(f'Total time: {total_time:.5f} secs/{(total_time/60):.5f} mins\nAverage time per epoch: {(total_time/n_epochs):.5f} seconds')
    plt.plot(history)
    
    model.load_state_dict(best_weights)
    return model

def save_model(model, filename='4_135_135.pth'):
    """
    After training the model, save it so we can use it later.
    """
    torch.save(model.state_dict(), filename)


def load_model(image_size=(4, 135, 135), filename='4_135_135.pth'):
    """
    Load the model from the saved state dictionary.
    """
    model = CNNRegression(image_size)
    model.load_state_dict(torch.load(filename))
    return model

def evaluate_network(model, device, image_size: Tuple[int, int, int] = (4, 135, 135)):
    """
    This evaluates the network on the test data.
    """
    if image_size[0] == 1:
        grayscale = True
    else:
        grayscale = False
    assert image_size[1] == image_size[2], 'Image size must be square'
    
    regression_task = RegressionTaskData(grayscale=grayscale)
    criterion = nn.MSELoss()
    max_loss = 0
    min_loss = np.inf
    # Evaluate the model on the test data
    with torch.no_grad():
        total_loss = 0
        total_distance_loss = 0
        n_samples_total = 0
        for inputs, targets in regression_task.testloader:
            # Calculate the loss with the criterion we used in training
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))
            total_loss += loss.item()

            # We are measuring the predicted distance away from the actual point, which is more useful to us than MSE loss
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            distance_losses = []
            for i in range(len(outputs_np)):
                deltax = np.abs(outputs_np[i][0]-targets_np[i][0])
                deltay = np.abs(outputs_np[i][1]-targets_np[i][1])

                distance = np.sqrt(deltax*deltax + deltay*deltay)
                distance_losses.append(distance)
            if distance < min_loss:
                min_loss = distance
            if distance > max_loss:
                max_loss = distance
            total_distance_loss += sum(distance_losses)
            n_samples_total += len(distance_losses)

        mean_loss = total_loss / len(regression_task.testloader)
        mean_distance_error = total_distance_loss / n_samples_total
        return mean_loss, min_loss, max_loss, mean_distance_error
        #print(f'Test Loss: {mean_loss:.4f}')
        #print(f'Test mean distance error: {mean_distance_error:.4f} meters')

def predict(rgbaimage, device = "cpu", filename = "2.7mm.pth"):
    model = load_model(filename = filename)
    model.to(device)
    tensorer = transforms.ToTensor()
    image = torch.from_numpy(np.array(tensorer(rgbaimage))).float()
    output = model(image.to(device))
    return output.detach().numpy()[0]
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Train the model
    image_size: Tuple[int, int, int] = (3, 100, 100)
    model = train_network(device, 20, image_size=image_size)

    # Save the model
    filename = f'{image_size[0]}_{image_size[1]}_{image_size[2]}.pth'
    save_model(model, filename=filename)

    # Load the model
    model = load_model(image_size=image_size, filename=filename)
    model.to(device)

    # Evaluate the model
    evaluate_network(model, device, image_size=image_size)
