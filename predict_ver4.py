#Refer to https://github.com/miguelangel/ai--transfer-learning-for-image-classification/blob/master/image_classifier.ipynb
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from train import load_model
import numpy as np

from PIL import Image
import os

import json
import argparse

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--topk', type=int, help='Return top K predictions')
parser.add_argument('--labels', type=str, help='JSON file that contains class values')
parser.add_argument('--device', action='store_true', help='Processing unit')
parser.add_argument('--image_path', type=str, help='Input the image path to predict')

label = 'cat_to_name.json'
with open(label, 'r') as f:
    cat_to_name = json.load(f)
    


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    normalize = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        normalize
    ])
    
    final_img = preprocess(image)

    return final_img

 # TODO: Implement the code to predict the class from an image file
topk = int(input("Please input the top K(num) to predict classes and probabilities:"))
label = input("Please load a JSON file:")
device = input("Please input which would you like to predict classes and probabilities: \ncpu \ngpu \n")
def predict(image_path, checkpoint, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    ''' 
    checkpoint = torch.load("checkpoint.pth")
    model=checkpoint['model']

    # Use gpu if selected and available
    if device == 'gpu':
        print("You are running on GPU! \nHere are the probabilities and classes:")
        torch.cuda.is_available()
        if True:
            model.cuda()
        else:
            print("Seems CUDA is not available. \nPlease check your CUDA installation correctly or not.")

    elif device == 'cpu':
        print("You are running on CPU! \nHere are the probabilities and classes:")
        torch.cuda.is_available()
        if True:
            model.to(device)
        else:
            print("Seems CUDA is not available. \nPlease check your CUDA installation correctly or not.")


    was_training = model.training    
    model.eval()
    
    pil_image = Image.open(image_path)
    pil_image = process_image(pil_image).float()
    
    image = np.array(pil_image)    
    
    image = Variable(torch.FloatTensor(image), requires_grad=True)
    image = image.unsqueeze(0) # this is for VGG
    
    if device == 'gpu':
        torch.cuda.is_available()
        if True:
            image = image.cuda()
        else:
            print("Seems CUDA is not available. \nPlease check your CUDA installation correctly or not.")

    elif device == 'cpu':
        torch.cuda.is_available()
        if True:
            image = image
        else:
            print("Seems CUDA is not available. \nPlease check your CUDA installation correctly or not.")

            
    result = model(image).topk(topk)
    
    if device == 'gpu':
        torch.cuda.is_available()
        if True:
            # Added softmax here as per described here:
            # # https://github.com/pytorch/vision/issues/432#issuecomment-368330817
            probs = torch.nn.functional.softmax(result[0].data, dim=1).cpu().numpy()[0]
            classes = result[1].data.cpu().tolist()
        else:
            print("Seems CUDA is not available. \nPlease check your CUDA installation correctly or not.")

    elif device == 'cpu':
        torch.cuda.is_available()
        if True:
            probs = torch.nn.functional.softmax(result[0].data, dim=1).numpy()[0]
            classes = result[1].tolist()
        else:
            print("Seems CUDA is not available. \nPlease check your CUDA installation correctly or not.")

    model.train(mode=was_training)
    
    return probs, classes

image_path=input("Please input the image_path you would like to predict:")
checkpoint="checkpoint.pth"
probas, classes = predict(image_path, checkpoint, topk)
print(probas)
print(classes)