import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import time
import numpy as np
import matplotlib.pyplot as plt
import json

arch = input("Enter the architecture ('VGG' or 'Densenet'): ")
gpu = False
epochs = 4
learning_rate = 0.001
hidden_units = 600   

def parse_args():
    parser = argparse.ArgumentParser(description="Train the network")
    parser.add_argument('data_dir',default='/content/drive/MyDrive/flowers', type=str,
                         help='enter the path to the directory containing the dataset for flower images. ')
    parser.add_argument('--save_dir', type=str, help ='enter directory for saved model', 
                        default = '/content/drive/MyDrive')
    parser.add_argument('--arch', default='vgg16', choices=['vgg16', 'densenet'],
                        help='enter deep NN architecture, options: vgg16, densenet')
    parser.add_argument('--gpu', default=False, type=bool,
                        help='flag for indicating whether or not to use a GPU in training')
    parser.add_argument('--epochs', type=int, help ='enter number of epochs ', default = 7)
    
    parser.add_argument('--learning_rate', type=float, help ='enter the learning rate', default = 0.001)
    
    parser.add_argument('--hidden_units', type=int, help ='enter number of hidden units ',default = 600)
    return parser.parse_args()

def printing_args(args):
    print("The information of our  vgg model:")
    print(f"The architecture of our model: {args.arch}")
    print(f"Using GPU: {args.gpu}")
    print(f"The learning rate for our model: {args.learning_rate}")
    print(f"Number of neurons in hidden layer: {args.hidden_units}")
    print(f"Number of training Epochs for the model: {args.epochs}")
    
def build_model(arch):    
    if arch == 'VGG':
        print("Using vgg16 pretrained model")
        model = models.vgg16(pretrained=True)
        input_size = 25088  

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, 4096)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(4096, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    elif arch == 'Densenet':
    print("Using Densenet pretrained model")
    model = models.densenet121(pretrained=True)
    input_size = 1024  

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, 512)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(512, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    return model
    
def train_model(model,criterion, optimizer, train_data ,valid_data,epochs,gpu):
    model = build_model(arch) 
    if gpu and torch.cuda.is_available():
        print("Using GPU")
        device='cuda'
    else:
         device='cpu'
   
    model.to(device)   
    steps=0
    print_every=10
    print("the beginning of training")
    start=time.time()
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_data):
            if gpu and torch.cuda.is_available():
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            steps += 1
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        
            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                
                with torch.no_grad():
                    for jj, (images, labels2) in enumerate(valid_data):
                        if gpu and torch.cuda.is_available():
                            images, labels2 = images.to('cuda'), labels2.to('cuda')
                        
                        outputs = model.forward(images)
                        valloss = criterion(outputs,labels2)
                        valid_loss += valloss.item()
                        ps = torch.exp(outputs)
                        equal = (labels2.data == ps.max(dim=1)[1])
                        accuracy += equal.type(torch.FloatTensor).mean()

                
                print(f"epoch No.{e+1}"
                    f"runnging loss: {running_loss/print_every:.3f}"
                    f"validation Loss: {valid_loss/len(valid_data):.3f}"
                    f"Accuracy: {accuracy/len(valid_data):.3f}")
            
                running_loss = 0
                model.train()
            
    total_time=time.time()-start 
    print("the total time is:{:.0f}m {:.0f}s".format(total_time//60, total_time % 60))

def save_checkpoint(path, model, optimizer, hidden_units, arch, learning_rate, epochs, classifier):
    checkpoint = {
        'hidden_units': hidden_units,
        'structure': arch,
        'learning_rate': learning_rate,
        'state_dict': model.state_dict(),
        'classifier': classifier,
        'num_epochs': epochs,
        'class_to_idx': model.class_to_idx,
        'opt_state': optimizer.state_dict()
    }
    torch.save(checkpoint, path + '/checkpoint.pth')

def main():
    print("The beginning")
    args=parse_args()
    printing_args(args)
    
    print(" preprocessing the data")
    data_dir= args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
    # TODO: Load the datasets with ImageFolder
image_datasets = {'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
}


    # TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),'valid': DataLoader(image_datasets['valid'], batch_size=batch_size),
'test': DataLoader(image_datasets['test'], batch_size=batch_size)
}
train_data = dataloaders ['train']
valid_data = dataloaders ['valid']  
print("Done preprocessing the data")
model=build_model(args.arch, args.hidden_units)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
print("Done building the model")
train_model(model,criterion,optimizer,train_data,valid_data,int(args.epochs),args.gpu)
model.class_to_idx = train_datasets.class_to_idx
print("saving the model")
save_checkpoint(args.save_dir, model, optimizer, args, model.classifier)
print("Done saving the model")

if __name__ == '__main__':
    main()
