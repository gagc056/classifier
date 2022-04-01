#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('wget https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/Positive_tensors.zip ')


# In[2]:


get_ipython().system('unzip -q Positive_tensors.zip ')


# In[3]:


get_ipython().system(' wget https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/Negative_tensors.zip')
get_ipython().system('unzip -q Negative_tensors.zip')


# In[4]:


get_ipython().system('pip install torchvision')


# In[5]:


#libraries
import torchvision.models as models
from PIL import Image
import pandas
from torchvision import transforms
import torch.nn as nn
import time
import torch 
import matplotlib.pylab as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
import os
import glob
torch.manual_seed(0)


# In[19]:


from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
from PIL import Image
import pandas as pd
import os


# In[27]:


pwd


# In[43]:


class Dataset(Dataset):

    # Constructor
    def __init__(self,transform=None,train=True):
        directory="/home/wsuser/work"
        positive="Positive_tensors"
        negative='Negative_tensors'

        positive_file_path=os.path.join(directory,positive)
        negative_file_path=os.path.join(directory,negative)
        positive_files=[os.path.join(positive_file_path,file) for file in os.listdir(positive_file_path) if file.endswith(".pt")]
        negative_files=[os.path.join(negative_file_path,file) for file in os.listdir(negative_file_path) if file.endswith(".pt")]
        number_of_samples=len(positive_files)+len(negative_files)
        self.all_files=[None]*number_of_samples
        self.all_files[::2]=positive_files
        self.all_files[1::2]=negative_files 
        # The transform is goint to be used on image
        self.transform = transform
        #torch.LongTensor
        self.Y=torch.zeros([number_of_samples]).type(torch.LongTensor)
        self.Y[::2]=1
        self.Y[1::2]=0
        
        if train:
            self.all_files=self.all_files[0:30000]
            self.Y=self.Y[0:30000]
            self.len=len(self.all_files)
        else:
            self.all_files=self.all_files[30000:]
            self.Y=self.Y[30000:]
            self.len=len(self.all_files)     
       
    # Get the length
    def __len__(self):
        return self.len
    
    # Getter
    def __getitem__(self, idx):
               
        image=torch.load(self.all_files[idx])
        y=self.Y[idx]
                  
        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y
    
print("done")


# In[44]:


train_dataset = Dataset(train=True)
validation_dataset = Dataset(train=False)
print("done")


# In[45]:


# transform data
model = models.resnet18(pretrained = True)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transforms.ToTensor()
transforms.Normalize(mean, std)
transforms.Compose([])

composed =transforms.Compose([ transforms.Resize(224),transforms.ToTensor(), transforms.Normalize(mean, std)])


# In[46]:


# Set the parameter cannot be trained for the pre-trained model
for param in model.parameters():
  param.requires_grad = False


# In[47]:


model.fc = nn.Linear(512, 2)


# In[48]:


print(model)


# In[49]:


#Create the loss function
criterion = nn.CrossEntropyLoss()


# In[50]:



batch_size = 100
trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size)
validloader = DataLoader(dataset=validation_dataset, batch_size=batch_size)


# In[51]:


optimizer = torch.optim.Adam([parameters  for parameters in model.parameters() if parameters.requires_grad],lr=0.001)


# In[55]:


n_epochs=1
loss_list=[]
accuracy_list=[]
correct=0
N_test=len(validation_dataset)
N_train=len(train_dataset)
start_time = time.time()
#n_epochs

Loss=0
start_time = time.time()
for epoch in range(n_epochs):
    for x, y in trainloader:

        model.train() 
        #clear gradient 
        optimizer.zero_grad()
        #make a prediction 
        z=model(x)
        # calculate loss 
        loss=criterion(z,y)
        # calculate gradients of parameters 
        loss.backward()
        # update parameters 
        optimizer.step()
        loss.data
        loss_list.append(loss.data)
    correct=0
    for x_test, y_test in validloader:
        # set model to eval 
        model.eval()
        #make a prediction 
        z=model(x_test)
        #find max 
        _,yhat=torch.max(z.data,1)
       
        #Calculate misclassified  samples in mini-batch 
        #hint +=(yhat==y_test).sum().item()
        correct+=(yhat==y_test).sum().item()
   
    accuracy=correct/N_test
    accuracy


# In[57]:


print(accuracy)


# In[58]:


plt.plot(loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()


# In[59]:


validloader = DataLoader(dataset=validation_dataset, batch_size=1)
model.eval()
counter = 0
indexes = []
idx = 0
for x, y in validloader:
    idx += 1
    z = model(x)
    _, yhat = torch.max(z.data, 1)
    if yhat != y:
      counter += 1
      indexes.append([idx, yhat, y])
    if counter > 3:
      break


# In[60]:


for i in indexes:
    print(f"Sample : {i[0]}, predicted value: {i[1]}, actual value: {i[2]}")


# In[ ]:




