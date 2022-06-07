import dataset
import model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('/home/zacharyyeh/Projects/ML2021Spring/HW2-1_Classification/runs')

#set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
#hyper parameters
n_epoch = 50
batch_size = 64
lr = 0.0001
weight_decay = 1e-4
MODEL_PATH = '/home/zacharyyeh/Projects/ML2021Spring/HW2-1_Classification/ckpt/Last.pth'
BEST_MODEL_PATH = '/home/zacharyyeh/Projects/ML2021Spring/HW2-1_Classification/ckpt/Best.pth'

#dataset and dataloader
print('loading datasets...')
train_set = dataset.TIMITDataset(mode="train")
val_set = dataset.TIMITDataset(mode="val")
train_loader = DataLoader(dataset = train_set, shuffle = True, batch_size = batch_size, num_workers = 2)
val_loader = DataLoader(dataset = val_set, shuffle = True, batch_size = batch_size, num_workers = 2)

#setup model
print('setting up model...')
model = model.Classifier().to(device)
criterion = nn.CrossEntropyLoss()
#to solve overfitting, we add regulization
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#training loop
print('start training...')
last_val_acc = 0
best_model_saved = False
for epoch in range(n_epoch):
    #reset acc and loss every epoch
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    model.train()
    for i, (features, labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)
        predicted = model(features)
        _, prediction = torch.max(predicted, 1)
        loss = criterion(predicted, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_acc += (prediction.cpu() == labels.cpu()).sum().item()
        train_loss += loss.item()

    writer.add_scalars('train/val accuracy', {'train_accuracy': train_acc/len(train_set)}, epoch+1)
    writer.add_scalars('train/val loss', {'train_loss': train_loss/len(train_set)}, epoch+1)
    print(f'epoch: {epoch+1}/{n_epoch}, train_acc: {train_acc/len(train_set):.3f}, train_loss: {train_loss/len(train_set):.3f}', end = ', ')

    #validation
    model.eval()
    with torch.no_grad():
        for i, (features, labels) in enumerate(val_loader):
            features = features.to(device)
            labels = labels.to(device)
            predicted = model(features)
            _, prediction = torch.max(predicted, 1)
            loss = criterion(predicted, labels)
            val_acc += (prediction.cpu() == labels.cpu()).sum().item()
            val_loss += loss.item()
 
    writer.add_scalars('train/val accuracy', {'val_accuracy' :val_acc/len(val_set)}, epoch+1)
    writer.add_scalars('train/val loss', {'val_loss': val_loss/len(val_set)}, epoch+1)
    print(f'val_acc:{val_acc/len(val_set):.3f}, val_loss: {val_loss/len(val_set):.3f}')

    #save best model
    if best_model_saved == False:
        if val_acc < last_val_acc+0.05:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print('Best model is saved to path:', BEST_MODEL_PATH)
            best_model_saved = True
    last_val_acc = val_acc

print('Training is completed')
torch.save(model.state_dict(), MODEL_PATH)
print('Model is saved to to path:', MODEL_PATH)











