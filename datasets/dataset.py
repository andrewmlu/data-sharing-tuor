import os
import numpy as np


def get_one_hot_from_label_index(label, number_of_labels=10):
    one_hot = np.zeros(number_of_labels)
    one_hot[label] = 1
    return one_hot


def normalize_data(X):
    for i in range(0, len(X[0,:])):
       X[:,i] = (X[:,i]-np.mean(X[:,i]))/np.std(X[:,i])


def get_even_odd_from_one_hot_label(label):
    for i in range(0, len(label)):
        if (label[i] == 1):
            c=i%2
            if c==0:
                c=1
            elif c==1:
                c=-1
            return c


def get_index_from_one_hot_label(label):
    for i in range(0, len(label)):
        if (label[i] == 1):
            return [i]


def get_data(dataset, total_data, dataset_file_path=os.path.dirname(__file__), sim_round=None, use_test_client_to_data_dict=False):
    test_client_to_data_dict = None
    print(dataset)

    if dataset=='MNIST_ORIG_EVEN_ODD' or dataset=='MNIST_ORIG_ALL_LABELS':
        from datasets.mnist_extractor import mnist_extract

        if total_data > 60000:
            total_data_train = 60000
        else:
            total_data_train = total_data

        if total_data > 10000:
            total_data_test = 10000
        else:
            total_data_test = total_data

        if sim_round is None:
            start_index_train = 0
            start_index_test = 0
        else:
            start_index_train = (sim_round * total_data_train) % (max(1, 60000 - total_data_train + 1))
            start_index_test = (sim_round * total_data_test) % (max(1, 10000 - total_data_test + 1))

        train_image, train_label = mnist_extract(start_index_train, total_data_train, True, dataset_file_path)
        test_image, test_label = mnist_extract(start_index_test, total_data_test, False, dataset_file_path)

        # train_label_orig must be determined before the values in train_label are overwritten below
        train_label_orig=[]
        for i in range(0, len(train_label)):
            label = get_index_from_one_hot_label(train_label[i])
            train_label_orig.append(label[0])

        if dataset == 'MNIST_ORIG_EVEN_ODD':
            for i in range(0, len(train_label)):
                train_label[i] = get_even_odd_from_one_hot_label(train_label[i])

        if dataset == 'MNIST_ORIG_EVEN_ODD':
            for i in range(0, len(test_label)):
                test_label[i] = get_even_odd_from_one_hot_label(test_label[i])


    elif dataset=='MNIST_FASHION':
        from datasets.mnist_extractor import mnist_extract

        if total_data > 60000:
            total_data_train = 60000
        else:
            total_data_train = total_data

        if total_data > 10000:
            total_data_test = 10000
        else:
            total_data_test = total_data

        if sim_round is None:
            start_index_train = 0
            start_index_test = 0
        else:
            start_index_train = (sim_round * total_data_train) % (max(1, 60000 - total_data_train + 1))
            start_index_test = (sim_round * total_data_test) % (max(1, 10000 - total_data_test + 1))

        train_image, train_label = mnist_extract(start_index_train, total_data_train, True, dataset_file_path, is_fashion_dataset=True)
        #print(np.array(train_image).shape)

        test_image, test_label = mnist_extract(start_index_test, total_data_test, False, dataset_file_path, is_fashion_dataset=True)

        train_label_orig=[]
        for i in range(0, len(train_label)):
            label = get_index_from_one_hot_label(train_label[i])
            train_label_orig.append(label[0])

    elif dataset == 'SVNH':
        train_image = np.load(dataset_file_path + '/SVNH/train_image_svnh.npy')
        train_label = np.load(dataset_file_path + '/SVNH/train_label_svnh_hot.npy')
        test_image = np.load(dataset_file_path + '/SVNH/test_image_svnh.npy')
        test_label = np.load(dataset_file_path + '/SVNH/test_label_svnh_hot.npy')
        train_label_orig = []

    elif dataset == 'CIFAR_10':
        from datasets.cifar_10_extractor import cifar_10_extract

        if total_data > 50000:
            total_data_train = 50000
        else:
            total_data_train = total_data

        if total_data > 10000:
            total_data_test = 10000
        else:
            total_data_test = total_data

        train_image, train_label = cifar_10_extract(0, total_data_train, True, dataset_file_path)
        test_image, test_label = cifar_10_extract(0, total_data_test, False, dataset_file_path)

        train_label_orig = []
        for i in range(0, len(train_label)):
            label = get_index_from_one_hot_label(train_label[i])
            train_label_orig.append(label[0])



    elif dataset == 'SVNH_GRAY':
        train_image = np.load(dataset_file_path + '/svnh-gray-28by28/svnh_train_image_gray.npy')
        train_label = np.load(dataset_file_path + '/svnh-gray-28by28/svnh_train_label_gray.npy')
        test_image = np.load(dataset_file_path + '/svnh-gray-28by28/svnh_test_image_gray.npy')
        test_label = np.load(dataset_file_path + '/svnh-gray-28by28/svnh_test_label_gray.npy')
        train_label_orig = []

    elif dataset == 'CIFAR_10_GRAY':
        train_image = np.load(dataset_file_path + '/cifar-gray-28by28/cifar_train_image_gray.npy')
        train_label = np.load(dataset_file_path + '/cifar-gray-28by28/cifar_train_label_gray.npy')
        test_image = np.load(dataset_file_path + '/cifar-gray-28by28/cifar_test_image_gray.npy')
        test_label = np.load(dataset_file_path + '/cifar-gray-28by28/cifar_test_label_gray.npy')
        train_label_orig = []

    elif dataset == 'MNIST_ORIG_ALL_LABELS_RGB':
        train_image = np.load(dataset_file_path + '/mnist_rgb_32by32/mnist_train_image_rgb.npy')
        train_label = np.load(dataset_file_path + '/mnist_rgb_32by32/mnist_train_label_rgb.npy')
        test_image = np.load(dataset_file_path + '/mnist_rgb_32by32/mnist_test_image_rgb.npy')
        test_label = np.load(dataset_file_path + '/mnist_rgb_32by32/mnist_test_label_rgb.npy')
        train_label_orig = []

    elif dataset == 'MNIST_FASHION_RGB':
        train_image = np.load(dataset_file_path + '/fashion_rgb_32by32/fashion_train_image_rgb.npy')
        train_label = np.load(dataset_file_path + '/fashion_rgb_32by32/fashion_train_label_rgb.npy')
        test_image = np.load(dataset_file_path + '/fashion_rgb_32by32/fashion_test_image_rgb.npy')
        test_label = np.load(dataset_file_path + '/fashion_rgb_32by32/fashion_test_label_rgb.npy')
        train_label_orig = []







    else:
        raise Exception('Unknown dataset name.')

    if use_test_client_to_data_dict:
        return train_image, train_label, test_image, test_label, train_label_orig, test_client_to_data_dict
    else:
        return train_image, train_label, test_image, test_label, train_label_orig


def get_data_train_samples(dataset, samples_list, dataset_file_path=os.path.dirname(__file__)):
    if dataset=='MNIST_ORIG_EVEN_ODD' or dataset=='MNIST_ORIG_ALL_LABELS':
        from datasets.mnist_extractor import mnist_extract_samples

        train_image, train_label = mnist_extract_samples(samples_list, True, dataset_file_path)

        if dataset == 'MNIST_ORIG_EVEN_ODD':
            for i in range(0, len(train_label)):
                train_label[i] = get_even_odd_from_one_hot_label(train_label[i])

    elif dataset=='MNIST_FASHION':
        from datasets.mnist_extractor import mnist_extract_samples

        train_image, train_label = mnist_extract_samples(samples_list, True, dataset_file_path, is_fashion_dataset=True)


    else:
        raise Exception('Training data sampling not supported for the given dataset name, use entire dataset by setting batch_size = total_data, ' +
                        'also confirm that dataset name is correct.')


    return train_image,train_label
