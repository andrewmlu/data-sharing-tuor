from torchvision.datasets import MNIST, CIFAR10, SVHN, CIFAR100,FashionMNIST
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
from torch.nn import functional


def get_single_dataloader(dataset, datadir, transform, size):
    train_bs = 10000
    test_bs = 10000

    if dataset == 'MNIST':
        dl_obj = MNIST
        mean = (0.1307, )
        std = (0.3081, )
        dim = 28
        outchannel = 1
    elif dataset == 'CIFAR10':
        dl_obj = CIFAR10
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        outchannel = 3
        dim = 32
    elif dataset == 'SVHN':
        dl_obj = SVHN
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.198, 0.201, 0.197]
        outchannel = 3
        dim = 32
    elif dataset == 'CIFAR100':
        dl_obj = CIFAR100
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        outchannel = 3
        dim = 32
    elif dataset == 'FASHION':
        dl_obj = FashionMNIST
        dim = 32

    else:
        raise Exception('Unknown dataset')

    if transform is None:

        transform = transforms.Compose([
            transforms.Resize((dim, dim)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


    elif transform == '3channel':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])
        outchannel = 3

    elif transform == '1channel':
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        outchannel = 1


    else:
        raise ('unknow transformation')

    if dataset == 'SVHN':
        train_ds = dl_obj(datadir,
                          split='train',
                          transform=transform,
                          target_transform=None,
                          download=True)
        test_ds = dl_obj(datadir,
                         split='test',
                         transform=transform,
                         target_transform=None,
                         download=True)

    else:

        train_ds = dl_obj(datadir,
                          train=True,
                          transform=transform,
                          download=True)
        test_ds = dl_obj(datadir,
                         train=False,
                         transform=transform,
                         download=True)


    train_dl = data.DataLoader(dataset=train_ds,
                               batch_size=train_bs,
                               shuffle=False)
    #print(len(train_dl))

    test_dl = data.DataLoader(dataset=test_ds,
                              batch_size=test_bs,
                              shuffle=False)

    return train_dl, test_dl



datadir = 'datasets'
train_dl, test_dl = get_single_dataloader('SVHN', datadir, '3channel', 32)
import matplotlib.pyplot as plt
# train_img = []
# for i, (inputs, labels) in enumerate(test_dl, 1):
#     t = np.array(inputs).shape[0]
#     img = np.array(inputs).reshape((t,32*32*3))
#     train_img.extend(img)
#     print(np.array(train_img).shape)
#
#
#
# np.save('dataset_files/SVNH/test_image_svhn.npy',train_img)

#
# train_label = []
# for i, (inputs, labels) in enumerate(train_dl, 1):
#     #t = np.array(labels).shape[0]
#     #label = np.array(inputs).reshape((t,28*28))
#     #train_img.extend(img)
#
#     a = np.array(functional.one_hot(labels, num_classes=10))
#     print(a.shape)
#     train_label.extend(a)
#
# print(np.array(train_label).shape)
# np.save('dataset_files/SVNH/train_label_svnh_hot.npy',train_label)
#
#
# # for cifar

# datadir = 'datasets'
# train_dl, test_dl = get_single_dataloader('CIFAR10', datadir, '1channel', 28)
#
# # train_img = []
# # for i, (inputs, labels) in enumerate(test_dl, 1):
# #     t = np.array(inputs).shape[0]
# #     img = np.array(inputs).reshape((t,28*28))
# #     train_img.extend(img)
# #     print(np.array(train_img).shape)
# # t = np.array(train_img)[0].reshape((28,28))
# # print(t.shape)
# # plt.imshow(t,cmap='gray')
# # plt.show()
# #
# # np.save('dataset_files/cifar-gray-28by28/cifar_test_image_gray.npy',train_img)
# #
#
#
#
# train_label = []
# for i, (inputs, labels) in enumerate(test_dl, 1):
#     #t = np.array(labels).shape[0]
#     #label = np.array(inputs).reshape((t,28*28))
#     #train_img.extend(img)
#
#     a = np.array(functional.one_hot(labels, num_classes=10))
#     print(a.shape)
#     train_label.extend(a)
#
# print(np.array(train_label).shape)
# #np.save('dataset_files/cifar-gray-28by28/cifar_train_label_gray.npy',train_label)
# np.save('dataset_files/cifar-gray-28by28/cifar_test_label_gray.npy',train_label)
#
#


# mnist
#datadir = 'datasets'
#train_dl, test_dl = get_single_dataloader('MNIST', datadir, '3channel', 32)

#
# train_img = []
# for i, (inputs, labels) in enumerate(test_dl, 1):
#     t= np.array(inputs).shape[0]
#     img = np.array(inputs).reshape((t,32*32*3))
#     train_img.extend(img)
#
# print(np.array(train_img).shape)
#
# #np.save('dataset_files/mnist_rgb_32by32/mnist_train_image_rgb.npy',train_img)
# np.save('dataset_files/mnist_rgb_32by32/mnist_test_image_rgb.npy',train_img)
#
#


# train_label = []
# for i, (inputs, labels) in enumerate(test_dl, 1):
#     #t = np.array(labels).shape[0]
#     #label = np.array(inputs).reshape((t,28*28))
#     #train_img.extend(img)
#
#     a = np.array(functional.one_hot(labels, num_classes=10))
#     print(a.shape)
#     train_label.extend(a)
#
# print(np.array(train_label).shape)
#
# #np.save('dataset_files/mnist_rgb_32by32/mnist_test_label_gray.npy',train_label)
#
# np.save('dataset_files/mnist_rgb_32by32/mnist_test_label_gray.npy',train_label)


# fashion


# datadir = 'datasets'
# train_dl, test_dl = get_single_dataloader('FASHION', datadir, '3channel', 32)

#
# train_img = []
# for i, (inputs, labels) in enumerate(test_dl, 1):
#     t= np.array(inputs).shape[0]
#     img = np.array(inputs).reshape((t,32*32*3))
#     train_img.extend(img)
#
# print(np.array(train_img).shape)
#
# #np.save('dataset_files/fashion_rgb_32by32/fashion_train_image_rgb.npy',train_img)
# np.save('dataset_files/fashion_rgb_32by32/fashion_test_image_rgb.npy',train_img)
#


#
# train_label = []
# for i, (inputs, labels) in enumerate(test_dl, 1):
#     #t = np.array(labels).shape[0]
#     #label = np.array(inputs).reshape((t,28*28))
#     #train_img.extend(img)
#
#     a = np.array(functional.one_hot(labels, num_classes=10))
#     print(a.shape)
#     train_label.extend(a)
#
# print(np.array(train_label).shape)
#
# #np.save('dataset_files/fashion_rgb_32by32/fashion_train_label_gray.npy',train_label)
#
# np.save('dataset_files/fashion_rgb_32by32/fashion_test_label_gray.npy',train_label)
