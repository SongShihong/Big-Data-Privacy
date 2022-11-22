import torch
import torch.optim as optim
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import  save_image
from torch.utils.data import Dataset, DataLoader
import random

class MyTrainSet(Dataset):
    def __init__(self, X = None, Y = None):
        # 定义好 image 的路径
        self.X, self.Y = X, Y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)
    
    def save(self, filename):
        torch.save([self.X, self.Y], filename)
    
    def load(self, filename):
        self.X, self.Y = torch.load(filename)

class TargetModel(nn.Module):
    def __init__(self) -> None:
        super(TargetModel, self).__init__()
        
        self.Flatten = nn.Flatten()
        self.Fc = nn.Sequential(
            nn.Linear(28 * 28, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )


    def forward(self, x):
        return self.Fc(self.Flatten(x))

    def fit(self, train_dataloader: DataLoader, epoch: int) -> list:
        '''
        Train the target model.

        Args:
            train_dataloader(DataLoader): Data loader of training dataset.
            epoch(int): Epoch of iterations.
        '''
        device = next(self.parameters()).device
        optimizer = optim.Adam(self.parameters())
        loss_function = nn.CrossEntropyLoss()
        loss_list = []
        pbar = tqdm.tqdm(range(epoch))
        for i in pbar:
            for train_data in train_dataloader:
                x = train_data[0].to(device)
                labels = train_data[1].to(device)
                outputs = self.forward(x)
                loss = loss_function(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            pbar.set_description(f'(Training target model)loss: {loss:.4f}')
                
            loss_list.append(loss.detach())
        return loss_list


class ShadowModel(nn.Module):
    def __init__(self) -> None:
        super(ShadowModel, self).__init__()
        
        self.Flatten = nn.Flatten()
        self.Fc = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.Fc(self.Flatten(x))

    def fit(self, train_dataloader: DataLoader, epoch: int) -> list:
        '''
        Train the target model.

        Args:
            train_dataloader(DataLoader): Data loader of training dataset.
            epoch(int): Epoch of iterations.
        '''
        device = next(self.parameters()).device
        optimizer = optim.Adam(self.parameters())
        loss_function = nn.CrossEntropyLoss()
        loss_list = []
        pbar = tqdm.tqdm(range(epoch))
        for i in pbar:
            for train_data in train_dataloader:
                x = train_data[0].to(device)
                labels = train_data[1].to(device)
                outputs = self.forward(x)
                loss = loss_function(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            pbar.set_description(f'(Training shadow model)loss: {loss:.4f}')
                
            loss_list.append(loss.detach())
        return loss_list

# GAN
class Generator(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(Generator, self).__init__()

        self.Flatten = nn.Flatten()
        self.fc_stack = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 784), 
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc_stack(self.Flatten(x))
    
class Discreminator(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(Discreminator, self).__init__()

        self.Flatten = nn.Flatten()
        self.fc_stack = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc_stack(self.Flatten(x))

class GAN(nn.Module):
    def __init__(self) -> None:
        super(GAN, self).__init__()
        self.G = Generator(100, 28 * 28)
        self.D = Discreminator(28 * 28, 1)
    
    def forward(self, x):
        return self.D(self.G(x))
    
    def fit(self, train_dataloader: DataLoader, epoch: int, lr = 0.0003) -> tuple:

        device = next(self.parameters()).device
        generator_optimer = optim.Adam(self.G.parameters(), lr = lr)
        discriminator_optimer = optim.Adam(self.D.parameters(), lr = lr)
        loss_function = nn.BCELoss()

        gen_loss = []
        dis_loss = []
        dis_acc_real = []
        dis_acc_fake = []
        gen_acc = []

        pbar = tqdm.tqdm(range(epoch))

        for i in pbar:
            for train_set in train_dataloader:
                
                # train the discriminator
                # use the real data
                real_data = train_set[0].detach().to(device)
                batch_size = real_data.size(0)
                real_data_labels = torch.ones(batch_size).to(device)
                real_outputs = self.D(real_data)
                d_loss_real = loss_function(real_outputs.view(-1), real_data_labels)
                # use the fake data
                fake_data = torch.randn(batch_size, 100).to(device)
                fake_outputs = self.forward(fake_data)
                fake_data_labels = torch.zeros(batch_size).to(device)
                d_loss_fake = loss_function(fake_outputs.view(-1), fake_data_labels)
                # optimize the discriminator's loss function
                discriminator_loss = d_loss_fake + d_loss_real
                discriminator_optimer.zero_grad()
                discriminator_loss.backward(retain_graph=False)
                discriminator_optimer.step()

                # train the generator
                fake_data = torch.randn(batch_size, 100).to(device)
                fake_outputs = self(fake_data)
                generator_loss = loss_function(fake_outputs.view(-1), real_data_labels)
                generator_optimer.zero_grad()
                generator_loss.backward(retain_graph=False)
                generator_optimer.step()
                
            pbar.set_description(f'(Training GAN)loss: {(discriminator_loss + generator_loss):.4f}')
            gen_loss.append(generator_loss)
            dis_loss.append(discriminator_loss)
            dis_acc_real.append(real_outputs.round().mean())
            dis_acc_fake.append(1 - fake_outputs.round().mean())
            gen_acc.append(fake_outputs.mean())
        
        return gen_loss, dis_loss, dis_acc_real, dis_acc_fake, gen_acc