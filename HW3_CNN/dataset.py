import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import model

dir_list = os.listdir('/home/zacharyyeh/Datasets/food-11/training/labeled')
train_dir = r'/home/zacharyyeh/Datasets/food-11/training/labeled'
val_dir = r'/home/zacharyyeh/Datasets/food-11/validation'

composed_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

class FoodDataset(Dataset):
    def __init__(self, mode = "train"):
        self.mode = mode
        if mode == "train":
            source_dir = train_dir
        elif mode == "val":
            source_dir = val_dir
        self.train_images = []
        self.train_labels = []
        #scan the training set for labels and images
        for i in range(len(dir_list)):
            dir_path = os.path.join(source_dir, dir_list[i])
            images = os.listdir(dir_path)
            for j in range(len(images)):
                image_path = os.path.join(dir_path, images[j])
                opened_image = Image.open(image_path)
                self.train_images.append(opened_image)
                self.train_labels.append(int(dir_list[i])%10 if int(dir_list[i])<10 else int(dir_list[i]))
        
    def __getitem__(self, index):
        img, label = self.train_images[index], self.train_labels[index]
        img, label = composed_transform(img), torch.tensor(label)
        return img, label
    
    def __len__(self):
        return len(self.train_images)

#create new dataset with pseudo labels
unlabeled_dir = r'/home/zacharyyeh/Datasets/food-11/training/unlabeled/00'
image_names = os.listdir(unlabeled_dir)
MODEL_PATH = 'model.pth'

if os.path.exists(MODEL_PATH):
    model = model.VGGNet()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

class SemiDataset(Dataset):
    def __init__(self):
        self.train_images = []
        #scan the training set for labels and images
        for i in range(len(image_names)):
            image_path = os.path.join(unlabeled_dir, image_names[i])
            opened_image = Image.open(image_path)
            self.train_images.append(opened_image)

    def __getitem__(self, index):
        img = self.train_images[index]
        img= composed_transform(img)
        img_for_model = img[None, :, :, :]
        prediction = model(img_for_model)
        _, label = torch.max(prediction, 1)
        return img, label[0]
    
    def __len__(self):
        return len(self.train_images)
    