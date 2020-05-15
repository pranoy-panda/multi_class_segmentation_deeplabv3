from model_folder.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
 
'''
First we get the pre-trained model from the torchvision function. 
Then we change the segmentation head. 
This is done by replacing the classifier module of the model with
a new DeepLabHead with new number of output channels
''' 
def createDeepLabv3(outputchannels=1):
    model = models.segmentation.deeplabv3_resnet101(
        pretrained=True, progress=True)
    # Added a Sigmoid activation after the last convolution layer
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    model.train()
    return model

