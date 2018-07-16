#Refer to https://github.com/miguelangel/ai--transfer-learning-for-image-classification/blob/master/image_classifier.ipynb
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
import torchvision.transforms as transforms

import argparse


# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--device', action='store_true', help='Processing unit')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--arch', type=str, help='Model architecture')
parser.add_argument('--learning_rate', type=float, help='Learning rate')



data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(224),
                                       transforms.CenterCrop(224),            
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])
    
test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),            
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])
    
# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

arch = input("Please choose a model you would like to train: \nvgg16 \nvgg19 \nalexnet\n")
def load_model(arch):
    #Let users to choose a training model.
    #Output should be a model what a user to choose
    global model
    if arch == "vgg16":
        model=models.vgg16(pretrained=True)
        #Freeze the parameters
        for param in model.parameters():
            param.requires_grad=False
        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout()),
                          ('fc2', nn.Linear(4096, 4096)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout()),
                          ('fc3', nn.Linear(4096, 102)),
                          ('logits', nn.LogSoftmax(dim=1))
                          ]))
        model.classifier=classifier
        return model
    
    elif arch == "vgg19":
        model=models.vgg19(pretrained=True)
        #Freeze the parameters
        for param in model.parameters():
            param.requires_grad=False
        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout()),
                          ('fc2', nn.Linear(4096, 4096)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout()),
                          ('fc3', nn.Linear(4096, 102)),
                          ('logits', nn.LogSoftmax(dim=1))
                          ]))
        model.classifier=classifier
        return model
    
    elif arch == "alexnet":
        model=models.alexnet(pretrained=True)
        #Freeze the parameters
        for param in model.parameters():
            param.requires_grad=False
        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(9216, 4096)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout()),
                          ('fc2', nn.Linear(4096, 4096)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout()),
                          ('fc3', nn.Linear(4096, 102)),
                          ('logits', nn.LogSoftmax(dim=1))
                          ]))
        model.classifier=classifier
        return model
    
load_model(arch)
    
# TODO: Train a model with a pre-trained network
#cuda = torch.device('cuda')
device = input("Please input which would you like to train model: \ncpu \ngpu \n")
learning_rate = input("Please input the learning rate you would like to train:")

if device == 'cpu':
  print("You are running on CPU!")
  torch.cuda.is_available()
  if True:
    device = torch.device("cuda")
    model.to(device)
  else:
    print("Seems CUDA is not available. \nPlease check your CUDA installation correctly or not.")
        

elif device == 'gpu':
  print("You are running on GPU!")
  torch.cuda.is_available()
  if True:
    device = torch.device("cuda:0")
    model.cuda()
  else:
    print("Seems CUDA is not available. \nPlease check your CUDA installation correctly or not.")



criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=int(float(learning_rate)))
arch = arch


# TODO: Do validation on the test set
def validation(model, validloader, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in validloader:

        #images.resize_(images.shape[0], 25088, 4096)

        images = images.cuda()
        output = model.forward(images)
        labels = labels.cuda()
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

epochs = int(input("How many epochs would you like to train the model?"))
steps = 0
running_loss = 0
print_every = 30
for e in range(epochs):
    model.train()
    for images, labels in trainloader:
        steps += 1
        
        optimizer.zero_grad()
        images = images.cuda()
        output = model.forward(images)
        labels = labels.cuda()
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            # Make sure network is in eval mode for inference
            model.eval()
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                test_loss, accuracy = validation(model, validloader, criterion)
                
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Validation Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
            
            running_loss = 0
            
            # Make sure training is back on
            model.train()

# Save the checkpoint
# Store class_to_idx into a model property
def save_checkpoint():
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'arch':arch, 'optimizer':optimizer, 'optimizer_dict':optimizer.state_dict(),
             'state_dict':model.state_dict(), 'epochs':e+1, 'class_to_idx':model.class_to_idx}
    torch.save(checkpoint, 'checkpoint.pth')
    return model


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    arch = checkpoint['arch']
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']
    for param in model.parameters():
        param.requires_grad = False 
    return model