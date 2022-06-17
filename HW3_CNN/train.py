import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dataset
import model

#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#declare parameters
input_size = 64*64
n_classes = 11
n_epoch = 20
batch_size = 100
learning_rate = 0.0001
semi = True
weight_decay = 1e-5


#Make dataset with transform included
trainset = dataset.FoodDataset(mode="train")
train_loader = DataLoader(dataset = trainset, batch_size = batch_size, shuffle = True)

if semi:
    semiset = dataset.SemiDataset()
    semi_loader = DataLoader(dataset = semiset, batch_size = batch_size, shuffle = True)
    trainset = torch.utils.data.ConcatDataset([trainset, semiset])
    train_loader = DataLoader(dataset=trainset, batch_size = batch_size, shuffle = True)

valset = dataset.FoodDataset(mode="val")
val_loader = DataLoader(dataset = valset, batch_size = batch_size, shuffle = False)

#Make model and loss funcitions
model = model.VGGNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)

#Make Training Loop
steps = len(train_loader)
print('Start training...')

for epoch in range(n_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.type(torch.long).to(device)
        
        #forward pass
        pred_labels = model(images)
        loss = criterion(pred_labels, labels)
        _, prediction = torch.max(pred_labels, 1)
        #backward pass
        loss.backward()

        #update gradients
        optimizer.step()
        optimizer.zero_grad()

        train_acc += (prediction.cpu() == labels.cpu()).sum().item()
        train_loss += loss.item()

        #print information in a epoch
        if (i+1) % 100 == 0:
            print(f'epoch: {epoch+1}/{n_epoch}, step: {(i+1)}/{steps}, loss: {loss.item():.3f}')

    print(f'epoch: {epoch+1}/{n_epoch}, train_acc: {train_acc/len(trainset):.3f}, train_loss: {train_loss/len(trainset):.3f}', end = ', ')

    #validation
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.type(torch.long).to(device)
            predicted = model(images)
            _, prediction = torch.max(predicted, 1)
            loss = criterion(predicted, labels)
            val_acc += (prediction.cpu() == labels.cpu()).sum().item()
            val_loss += loss.item()

    print(f'val_acc:{val_acc/len(valset):.3f}, val_loss: {val_loss/len(valset):.3f}')

print('Training is completed')

if not semi:
    torch.save(model.state_dict(), dataset.MODEL_PATH)
    print('model saved as:', dataset.MODEL_PATH)
else:
    torch.save(model.state_dict(), 'semi_model.pth')
    print('model saved as:', 'semi_model.pth')
