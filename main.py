import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score
from model import createDeepLabv3
from trainer import train_model
import datahandler
import argparse
import os
import torch

"""
    Version requirements:
        PyTorch Version:  1.2.0
        Torchvision Version:  0.4.0a0+6b959ee
"""

# Command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument(
    "data_directory", help='Specify the dataset directory path')
parser.add_argument(
    "exp_directory", help='Specify the experiment directory where metrics and model weights shall be stored.')
parser.add_argument("--epochs", default=25, type=int)
parser.add_argument("--batchsize", default=4, type=int)

args = parser.parse_args()


bpath = args.exp_directory
data_dir = args.data_directory
epochs = args.epochs
batchsize = args.batchsize
# Create the deeplabv3 resnet101 model which is pretrained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
model = createDeepLabv3()
model.train()
# Create the experiment directory if not present
if not os.path.isdir(bpath):
    os.mkdir(bpath)


# Specify the loss function
criterion = torch.nn.CrossEntropyLoss()
#criterion = torch.nn.BCEWithLogitsLoss()
#criterion = torch.nn.functional.cross_entropy()
# Specify the optimizer with a lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Specify the evalutation metrics
metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

# class id's
class_id_list = [0,1,2,3,4,5,6,7,8]

mapping = { # 8-Class eTRIMS Dataset (R G B)
        0: (0, 0, 128),       # Building
        1: (128, 0, 128),     # Car	
        2: (0, 128, 128),     # Door
        3: (128, 128, 128),   # Pavement
        4: (0, 64, 128),      # Road
        5: (128, 128, 0),     # Sky
        6: (0, 128, 0),       # Vegetation
        7: (128, 0, 0),       # Window
        8: (0, 0, 0)          # Void
    }
'''
mapping = { # 8-Class eTRIMS Dataset (R G B)
        (0, 0, 128):0,       # Building
        (128, 0, 128):1,     # Car	
        (0, 128, 128):2,     # Door
        (128, 128, 128):3,   # Pavement
        (0, 64, 128):4,      # Road
        (128, 128, 0):5,     # Sky
        (0, 128, 0):6,       # Vegetation
        (128, 0, 0):7,       # Window
        (0, 0, 0):8          # Void
    }

mask = cv2.imread(mask_name)
m = np.zeros((mask.shape[0],mask.shape[1]))
for x in range(mask.shape[0]):
	for y in range(mask.shape[1]):
		for k in mapping:
			if np.all(mask[x,y,:]==k):
				m[x,y] = mapping[k]

(very slow process)	
'''

# Create the dataloader
dataloaders = datahandler.get_dataloader_single_folder(
    data_dir, batch_size=batchsize, class_id_list=class_id_list,mapping=mapping)
trained_model = train_model(model, criterion, dataloaders,
                            optimizer, bpath=bpath, metrics=metrics, num_epochs=epochs)


# Save the trained model
# torch.save({'model_state_dict':trained_model.state_dict()},os.path.join(bpath,'weights'))
torch.save(model, os.path.join(bpath, 'weights.pt'))