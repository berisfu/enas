#coding=utf-8
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
# import cPickle as pickle
import os
import pickle
import time

import numpy as np


def load_data(path, data_types, flag):
    scale_size = 224
    size = 224
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(scale_size),
            transforms.CenterCrop(size),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            # transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(scale_size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(scale_size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    datas = {}
    imgs, labels = {}, {}
    data_size = {}
    data_loaders = {}
    for dt in data_types:
        print('{} data'.format(dt))
        start = time.time()
        data_path = os.path.join(path, dt)
        datas[dt] = ImageFolder(data_path, data_transforms[dt])
        data_size[dt] = len(datas[dt])
        print('Generating DataLoader...')
        data_loaders[dt] = DataLoader(
            datas[dt], batch_size=100, shuffle=True, num_workers=4)

        print('Enumerating imgs in batch...')
        for step, (batch_x, batch_y) in enumerate(data_loaders[dt]):
            if step == 0:
                X = batch_x.numpy()
                Y = batch_y.numpy()
            else:
                X = np.vstack((X, batch_x.numpy()))
                Y = np.append(Y, batch_y.numpy())

        imgs[dt]   = np.transpose(X, [0, 2, 3, 1]).astype('float32')
        labels[dt] = Y.astype('int32')
        if flag == 1:
            # save to pkl file for next time to load data
            imgs_and_labels = (imgs[dt], labels[dt])

            with open(path+"{}_skin_size_{}.pkl".format(dt,size), 'wb') as f:
                pickle.dump(imgs_and_labels, f, protocol=2)
        print('{}: imgs:{}, labels:{} costs {} seconds'.format(dt, imgs[dt].shape, labels[dt].shape, time.time()-start))
    return imgs, labels


def read_data(data_path, num_valids=5000):
    data_types = ['train', 'valid', 'test']
    print("-" * 80)
    print("Reading data")
    size = 224

    flag = 0 # read img files
    # flag = 1 # read pkl files
    if flag == 0:
        print('Loading data from imgs ...')
        return load_data(data_path, data_types, flag)
    elif flag == 1:
        if 'train_skin_size_{}.pkl'.format(size) not in os.listdir(data_path):
            # if *.pkl files not created
            print('First save imgs to pkl files, then load data ...')
            return load_data(data_path, data_types, flag)
        else:
            # if *.pkl files already exist
            print('Loading data from pkl files...')
            imgs,labels = {}, {}
            for dt in data_types:
                start = time.time()
                with open(data_path + "{}_skin_size_{}.pkl".format(dt,size), 'rb') as f:
                    imgs[dt], labels[dt] = pickle.load(f)
                    print('{} images.shape:{}, labels.shape:{} costs {} seconds'.format(dt,imgs[dt].shape, labels[dt].shape, time.time()-start))
            return imgs, labels

if __name__ == '__main__':
    imgs, labels = read_data('./data/skin5')
    for key in imgs:
        print("{0} imgs_{0}.shape={1}, labels_{0}.shape={2}".format(key,imgs[key].shape, labels[key].shape))
