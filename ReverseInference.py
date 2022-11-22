import torch
import torch.optim as optim
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import  save_image, make_grid
from torch.utils.data import Dataset, DataLoader
import random
import MyModels
import argparse

def GetAccuracy(
    model: nn.Module, 
    test_loader: DataLoader, 
    device: torch.device = torch.device('cuda')
) -> tuple[int, int]:
    '''
    Get accuracy of a model on test dataloader.

    Args:
        model(Moudle): The model to be tested.
        test_loader(DataLoader): The data loader of test data.
        device(device): device of model and test_loader.
    
    Returns:
        Tuple[int, int]:
            correct(int): number of correct data.
            tot(int): number of total data.
    '''

    tot = 0
    correct = 0
    for test_data in test_loader:
        x = test_data[0].to(device)
        labels = test_data[1].to(device)
        tot = tot + x.size()[0]
        # print((model(x) == label).sum())
        correct = correct + (model(x).max(1)[-1] == labels).sum()
    return correct, tot


def GenerateRealDataset(
    batch_size: int = 128,
    data_filepath: str = './data/',
    train_ratio: float = 0.8
) -> tuple[MyModels.MyTrainSet, MyModels.MyTrainSet]:
    '''
    Generate the real data loader of real dataset, which is used to train the target model.

    Args:
        batch_size(int): Size of batch.
        data_filepath(str): File path of training data.
        train_test_ratio(float): Ratio of size of train data and total data.
        real_trainset_savepath(str): File path of training dataset to be saved.
        real_testset_savepath(str): File path of test dataset to be saved.
    
    Returns:
        Tuple[MyTrainSet, MyTrainSet]:
            real_train_set(MyTrainSet): Dataset for training of real data.
            real_test_set(MyTrainSet): Dataset for test of real data.
    '''

    img_transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])
    mnist_train=datasets.MNIST(root=data_filepath, train=True,transform=img_transform,download=True)
    training_loader=torch.utils.data.DataLoader(dataset=mnist_train,batch_size = batch_size,shuffle=True)
    # mnist_test=datasets.MNIST(root=data_filepath, train=False,transform=img_transform,download=True)
    # test_loader=torch.utils.data.DataLoader(dataset=mnist_train,batch_size=test_batch_size,shuffle=True)
    
    # 筛选数据集
    # 比如筛选0和1的数据
    selected_data_list = torch.tensor([], dtype=torch.float)
    selected_label_list = torch.tensor([], dtype=torch.long)
    pbar = tqdm.tqdm(training_loader)
    pbar.set_description('Generating the real dataset')
    for data_set in pbar:
        x = data_set[0]
        y = data_set[1]
        zero_data_index = torch.nonzero(y == 0).view(-1)
        zero_data_set = x[zero_data_index]
        one_data_index = torch.nonzero(y == 1).view(-1)
        one_data_set = x[one_data_index]
        selected_data_list = torch.cat([selected_data_list, zero_data_set, one_data_set], 0)
        selected_label_list = torch.cat([selected_label_list, torch.zeros_like(zero_data_index), torch.ones_like(one_data_index)], 0)

    tot_num = len(selected_label_list)
    # print('total number of real training dataset is {}'.format(tot_num))

    real_train_set = MyModels.MyTrainSet(selected_data_list[: int(tot_num * train_ratio)], selected_label_list[: int(tot_num * train_ratio)])
    real_test_set = MyModels.MyTrainSet(selected_data_list[int(tot_num * train_ratio):], selected_label_list[int(tot_num * train_ratio):])
    # real_train_set.save(real_trainset_savepath)
    # real_test_set.save(real_testset_savepath)

    return real_train_set, real_test_set

def SampleDataset(
    big_dataset: Dataset,
    ratio: float
) -> MyModels.MyTrainSet:
    length = len(big_dataset)
    seeds = torch.rand([length])
    select_index = torch.nonzero(seeds < ratio).view(-1)
    selected = big_dataset[select_index]
    res = MyModels.MyTrainSet(selected[0], selected[1])
    return res
    

def GetHighConfidenceSet(
    target_model: nn.Module,
    big_dataset: MyModels.MyTrainSet,
    shreshold: float,
    label = None,
    device: torch.device = torch.device('cuda')
) -> MyModels.MyTrainSet:

    training_loader = DataLoader(big_dataset, batch_size=128, shuffle=True)
    attacker_selected_data_list = torch.tensor([], dtype=torch.float).to(device)
    attacker_selected_label_list = torch.tensor([], dtype=torch.long).to(device)

    pbar = tqdm.tqdm(training_loader)
    pbar.set_description('Generating attacker\'s training dataset.')
    for data_set in pbar:
        x = data_set[0].to(device)
        if not label:
            y = data_set[1].to(device)
        else:
            y = label * torch.ones(x.size()[0], dtype = torch.long, device = device)
        mask = nn.Softmax(1)
        out = target_model(x)
        selected_index = torch.nonzero((mask(out).max(1)[0] > shreshold) * (out.max(1)[-1] == y)).view(-1)
        attacker_selected_data_list = torch.cat([attacker_selected_data_list, x[selected_index]], 0)
        attacker_selected_label_list = torch.cat([attacker_selected_label_list, y[selected_index]], 0)
    attacker_train_set = MyModels.MyTrainSet(attacker_selected_data_list, attacker_selected_label_list)

    return attacker_train_set

def GetHighConfidenceData(
    target_model: nn.Module,
    x: torch.Tensor,
    shreshold: float,
    label = None
) -> torch.Tensor:
    
    mask = nn.Softmax(1)
    out = target_model(x)
    return x[torch.nonzero((mask(out)[:, label]).view(-1) > shreshold).view(-1), :]


def AddNoise(data: torch.Tensor, sigma: float):
    noise = torch.randn_like(data)
    return noise * sigma + data


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', type=float, default=0.1)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=2000)
    parser.add_argument('--size', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    args = parser.parse_args()
    
    device = torch.device('cuda')
    real_training_set, real_test_set = GenerateRealDataset()
    d = real_training_set.X[0: 16]
    with torch.no_grad():
        grid = make_grid(d, 4)
        save_image(grid, 'real_dataset.png')
    
    target_model = MyModels.TargetModel().to(device)
    real_training_dataloader = DataLoader(real_training_set, batch_size=128, shuffle=True)
    real_test_dataloader = DataLoader(real_test_set, batch_size=128, shuffle=True)
    target_model.fit(real_training_dataloader, 100)
    cor, tot = GetAccuracy(target_model, real_test_dataloader)
    print('real model arrcuacy:')
    print(cor, tot, cor / tot)
    
    attacker_dataset = SampleDataset(real_training_set, args.ratio)
    attacker_dataset.X = AddNoise(attacker_dataset.X, args.noise)
    d = attacker_dataset.X[0: 16]
    with torch.no_grad():
        grid = make_grid(d, 4)
        save_image(grid, 'dirty_and_small_attacker_dataset.png')
        
    chosen = GetHighConfidenceSet(target_model, attacker_dataset, 0.9)
    d = chosen.X[0:16]
    with torch.no_grad():
        grid = make_grid(d, 4)
        save_image(grid, 'generative_data_init.png')
    S = [0.8, 0.85, 0.9, 0.9]
    for i, s in enumerate(S):
        cd = DataLoader(chosen, batch_size=128, shuffle=True)
        gan = MyModels.GAN().to(device)
        gan.fit(cd, args.epoch, lr = args.lr)
        x = torch.randn([args.size, 100], device = device)
        out = gan.G(x)
        x0 = GetHighConfidenceData(target_model, out, s, 0)
        x1 = GetHighConfidenceData(target_model, out, s, 1)
        x = torch.cat([x0, x1], dim = 0)
        y = target_model(x).max(1)[-1]
        chosen = MyModels.MyTrainSet(x, y)
        d = x0[0:16].reshape([16, 1, 28, 28])
        with torch.no_grad():
            grid = make_grid(d, 4)
            save_image(grid, 'generative_data_{}_0.png'.format(i))
        d = x1[0:16].reshape([16, 1, 28, 28])
        with torch.no_grad():
            grid = make_grid(d, 4)
            save_image(grid, 'generative_data_{}_1.png'.format(i))
        torch.save(gan.state_dict(), 'models/gan_{}.pkt'.format(i))
    
    