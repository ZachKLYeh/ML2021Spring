import dataset
import model
import matplotlib.pyplot 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import seaborn as sn
from sklearn.metrics import confusion_matrix
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('/home/zacharyyeh/Projects/ML2021Spring/HW2-1_Classification/runs')

#set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#hyper parameters
MODEL_PATH = '/home/zacharyyeh/Projects/ML2021Spring/HW2-1_Classification/ckpt/Last.pth'
BEST_MODEL_PATH = '/home/zacharyyeh/Projects/ML2021Spring/HW2-1_Classification/ckpt/Best.pth'

#dataset and dataloader
print('loading datasets...')
test_set = dataset.TIMITDataset(mode="test")
test_loader = DataLoader(dataset = test_set, shuffle = True, batch_size = len(test_set), num_workers = 2)

#setup model
print('setting up model...')
best_model = model.Classifier().to(device)
best_model.load_state_dict(torch.load(BEST_MODEL_PATH))
best_model.eval()
'''
last_model = model.Classifier().to(device)
last_model.load_state_dict(torch.load(MODEL_PATH))
last_model.eval()

#testing last model
test_acc = 0.0
with torch.no_grad():
    for i, (features, labels) in enumerate(test_loader):
        features = features.to(device)
        labels = labels.to(device)
        predicted = last_model(features)
        _, prediction = torch.max(predicted, 1)
        test_acc += (prediction.cpu() == labels.cpu()).sum().item()
print(f'Last model accuracy:{test_acc/len(test_set):.3f}')
'''
#testing best model
#using best model to evaluate pr curve and confution matrix
test_acc = 0.0
with torch.no_grad():
    cf_pred = []
    cf_gt = []
    pr_prob = []
    pr_gt = []
    for i, (features, labels) in enumerate(test_loader):
        features = features.to(device)
        labels = labels.to(device)
        predicted = best_model(features)
        _, prediction = torch.max(predicted, 1)
        test_acc += (prediction.cpu() == labels.cpu()).sum().item()
        #include pr curve info
        pr_prob_batch = [nn.functional.softmax(outputs, dim=0) for outputs in predicted]
        pr_gt.append(labels)
        pr_prob.append(pr_prob_batch)
        #include cf matrix info
        prediction = prediction.cpu().numpy()
        cf_pred.extend(prediction)
        labels = labels.cpu().numpy()
        cf_gt.extend(labels)

print(f'Best model accuracy:{test_acc/len(test_set):.3f}')

#evaluate pr_cruve
#pr_cruve is evaluated via whether the prediction is correct and its predicted probability
pr_prob = torch.cat([torch.stack(batch) for batch in pr_prob]) 
pr_gt = torch.cat(pr_gt) 

prob = []
hit = []
for i in range(39):
    hit_i = pr_gt == i #transfrom a list of labels to a list of boolin
    prob_i = pr_prob[:, i] #given the class predicted probability
    hit.append(hit_i)
    prob.append(prob_i)
    
writer.add_pr_curve('PR curve', torch.cat(hit) ,torch.cat(prob), global_step=0)

#evaluate cf matrix
cf_matrix = confusion_matrix(cf_gt, cf_pred)
#include dataframe showing classes of prediction
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix, axis = 1), index = [i for i in range(39)], columns = [i for i in range(39)])
matplotlib.pyplot.figure(figsize=(50, 50))  
cf_fig = sn.heatmap(df_cm, annot = True).get_figure()
writer.add_figure('Confution Matrix', cf_fig)

writer.close()















