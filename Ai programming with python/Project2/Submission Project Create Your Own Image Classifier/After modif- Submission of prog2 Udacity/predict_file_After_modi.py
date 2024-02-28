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
from train_file_After_modi import build_model

def get_args():
    parser = argparse.ArgumentParser(description="Make predictions")
    parser.add_argument("--input_image", type=str, help="path to the input image",default='/content/drive/MyDrive/flowers/test/1/image_06743.jpg')
    parser.add_argument("--checkpoint_path", type=str, help="path to the checkpoint file", default='/content/drive/MyDrive/checkpoint.pth')
    parser.add_argument("--top_k", type=int, default=5, help="number of top predictions to return")
    parser.add_argument("--category_names", type=str, default='/content/cat_to_name.json', help="path to the JSON file containing the label to category mapping")
    parser.add_argument("--use_gpu", action='store_true', default=False, help="flag for indicating whether or not to use a GPU")
    parser.add_argument("--arch", type=str, choices=['vgg16', 'resnet18'], default='vgg16', help="choose between vgg16 and resnet18")
    return parser.parse_args()
    return model

def load_checkpoint(filepath, arch):
    checkpoint = torch.load(filepath)

    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
    else:
        raise ValueError("Unsupported architecture. Choose between 'vgg16' and 'resnet18'.")

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def load_cat_names(filename):
    with open(filename) as f:
        category_names = json.load(f)
    return category_names

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image)
    image = image.resize((256, 256))
    image = image.crop((16, 16, 240, 240))  # Crop to 224x224 at the center
    np_image = np.array(image) / 255.0  # Normalize to the range [0, 1]
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means) / stds
    np_image = np_image.transpose((2, 0, 1))
    image_tensor = torch.from_numpy(np_image).float()
    return image_tensor

def predict(image, model,use_gpu, topk=5):
    if use_gpu and torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"
    model.to(device)    
    model.eval()
    image = image.unsqueeze_(0)
    image = image.to(device)
    with torch.no_grad():
        output = model.forward(image)   
    ps = torch.exp(output)    
    top_k, top_classes_idx = ps.topk(topk, dim=1)
    top_k, top_classes_idx = np.array(top_k.to('cpu')[0]), np.array(top_classes_idx.to('cpu')[0])
    idx_to_class = {x: y for y, x in model.class_to_idx.items()}
    top_classes = []
    for index in top_classes_idx:
        top_classes.append(idx_to_class[index])
    
    return list(top_k), list(top_classes)

def main():
    print("The beginning")
    args=get_args()
    cat_to_name = load_cat_names(args.category_names)
    model = load_checkpoint(args.checkpoint_path, args.arch)
    print("\n Done building the model")
    processed_img=process_image(args.input_image)
    print("\n Done preprocess the image")
    top_p,top_classes=predict(processed_img, model, args.use_gpu,args.top_k)
    print("\n Done prediction!")
    labels = [cat_to_name[str(i)] for i in top_classes]
    print(f"Top {args.top_k} predictions are : {list(zip(labels, top_p))}")

if __name__ == '__main__':
    main()
