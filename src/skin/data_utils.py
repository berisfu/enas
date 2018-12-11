#coding=utf-8
import os
import pickle
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

def _read_cifar10_data(data_path, train_files):
  """Reads CIFAR-10 format data. Always returns NHWC format.

  Returns:
    images: np tensor of size [N, H, W, C]
    labels: np tensor of size [N]
  """
  images, labels = [], []
  for file_name in train_files:
    print (file_name)
    full_name = os.path.join(data_path, file_name)
    with open(full_name, 'rb') as finp: # for python3
      u = pickle._Unpickler(finp)
      u.encoding = 'latin1'
      data = u.load()
      batch_images = data["data"].astype(np.float32) / 255.0
      batch_labels = np.array(data["labels"], dtype=np.int32)
      images.append(batch_images)
      labels.append(batch_labels)
    # with open(full_name) as finp: # for python2
    #   data = pickle.load(finp) 
    #   batch_images = data["data"].astype(np.float32) / 255.0
    #   batch_labels = np.array(data["labels"], dtype=np.int32)
    #   images.append(batch_images)
    #   labels.append(batch_labels)
  images = np.concatenate(images, axis=0)
  labels = np.concatenate(labels, axis=0)
  images = np.reshape(images, [-1, 3, 32, 32])
  images = np.transpose(images, [0, 2, 3, 1])

  return images, labels


def load_data(path, img_size, data_types, flag):
    '''
    Reads folder image format data.
    '''
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            # transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
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

            with open(path+"{}_skin_size_{}.pkl".format(dt,img_size), 'wb') as f:
                pickle.dump(imgs_and_labels, f, protocol=2)
        print('{}: imgs:{}, labels:{} costs {} seconds'.format(dt, imgs[dt].shape, labels[dt].shape, time.time()-start))
    return imgs, labels


def read_data(data_path, img_size=224, num_valids=5000):
    data_types = ['train', 'valid', 'test']
    print("-" * 80)
    print("Reading data")

    if 'cifar10' in data_path:
        flag = 3
    else:
        flag = 0 # read img files
        # flag = 1 # read pkl files
    if flag == 0:
        print('Loading data from imgs ...')
        return load_data(data_path, img_size, data_types, flag)
    elif flag == 1:
        if 'train_skin_size_{}.pkl'.format(img_size) not in os.listdir(data_path):
            # if *.pkl files not created
            print('First save imgs to pkl files, then load data ...')
            return load_data(data_path, img_size, data_types, flag)
        else:
            # if *.pkl files already exist
            print('Loading data from pkl files...')
            imgs,labels = {}, {}
            for dt in data_types:
                start = time.time()
                with open(data_path + "{}_skin_size_{}.pkl".format(dt,img_size), 'rb') as f:
                    imgs[dt], labels[dt] = pickle.load(f)
                    print('{} images.shape:{}, labels.shape:{} costs {} seconds'.format(dt,imgs[dt].shape, labels[dt].shape, time.time()-start))
            return imgs, labels
    elif flag == 3:
        images, labels = {}, {}

        train_files = [
            "data_batch_1",
            "data_batch_2",
            "data_batch_3",
            "data_batch_4",
            "data_batch_5",
        ]
        test_file = [
            "test_batch",
        ]
        images["train"], labels["train"] = _read_cifar10_data(data_path, train_files)

        if num_valids:
            images["valid"] = images["train"][-num_valids:]
            labels["valid"] = labels["train"][-num_valids:]

            images["train"] = images["train"][:-num_valids]
            labels["train"] = labels["train"][:-num_valids]
        else:
            images["valid"], labels["valid"] = None, None

        images["test"], labels["test"] = _read_cifar10_data(data_path, test_file)

        print ("Prepropcess: [subtract mean], [divide std]")
        mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
        std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

        print ("mean: {}".format(np.reshape(mean * 255.0, [-1])))
        print ("std: {}".format(np.reshape(std * 255.0, [-1])))
        
        images["train"] = (images["train"] - mean) / std
        if num_valids:
            images["valid"] = (images["valid"] - mean) / std
        images["test"] = (images["test"] - mean) / std

        return images, labels



if __name__ == '__main__':
    imgs, labels = read_data('./data/skin5')
    for key in imgs:
        print("{0} imgs_{0}.shape={1}, labels_{0}.shape={2}".format(key,imgs[key].shape, labels[key].shape))
