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

    if dataset=='IRIS':
        from xlrd import open_workbook

        if total_data > 150:
            total_data_train = 150
            total_data_test = 150
        else:
            total_data_train = total_data
            total_data_test = total_data

        wb = open_workbook(dataset_file_path + '/irisdata.xlsx')
        values = np.zeros((total_data_train, 4))
        labelsAll = []
        for s in wb.sheets():
            for j in range(0, 150):
                for i in range(0, 4):
                    values[j][i] = (s.cell(j, i).value)

        imgsAll = values

        for s in wb.sheets():
            for j in range(0, total_data_test):
                labelsAll.append((s.cell(j, 4).value))

        labelsAll=[int(x) for x in labelsAll]

        train_image=imgsAll
        test_image=imgsAll
        train_label=labelsAll
        test_label=labelsAll
        train_label_orig=labelsAll

    elif dataset == 'USER_MODEL':
        from xlrd import open_workbook

        wb = open_workbook(dataset_file_path + '/dataset_user_modeling.xlsx')
        X = np.zeros((402, 6))

        for s in wb.sheets():

            for j in range(0, 402):

                for i in range(0, 6):

                    X[j][i] = (s.cell(j, i).value)

        # X= preprocessing.scale(X)

        y = []

        for j in range(0, 402):
            y.append(X[j][5])
        X=X[0:402,0:5]

        train_image = X
        test_image = X
        train_label =y
        test_label = y
        train_label_orig = y

    elif dataset=='SEGMENT':
        from xlrd import open_workbook

        cut = 1500

        wb = open_workbook(dataset_file_path + '/image_segmentation_dataset.xlsx')

        values = np.zeros((2310, 19))

        for s in wb.sheets():
            for j in range(0, 2310):
                for i in range(1, 20):
                    values[j][i - 1] = (s.cell(j, i).value)

        normalize_data(values)

        labelsStringInterm = []
        for s in wb.sheets():
            for j in range(0, 2310):
                labelsStringInterm.append((s.cell(j, 0).value))

        indexShuffled = np.random.permutation(len(labelsStringInterm))

        labelsString = []
        imgsAll = []

        for i in indexShuffled:
            labelsString.append(labelsStringInterm[i])
            imgsAll.append(values[i])

        # Transform label string in label digits
        labelOrigin = []
        labelBinary = []

        for i in range(0, len(labelsString)):
            if labelsString[i] == 'BRICKFACE':
                labelOrigin.append(1)
                labelBinary.append(1)
            elif labelsString[i] == 'SKY':
                labelOrigin.append(2)
                labelBinary.append(-1)
            elif labelsString[i] == 'FOLIAGE':
                labelOrigin.append(3)
                labelBinary.append(-1)
            elif labelsString[i] == 'CEMENT':
                labelOrigin.append(4)
                labelBinary.append(1)
            elif labelsString[i] == 'WINDOW':
                labelOrigin.append(5)
                labelBinary.append(1)
            elif labelsString[i] == 'PATH':
                labelOrigin.append(6)
                labelBinary.append(1)
            elif labelsString[i] == 'GRASS':
                labelOrigin.append(7)
                labelBinary.append(-1)



        train_image = []
        test_image = []
        train_label = []
        test_label = []
        train_label_orig = []
        for i in range(0, cut):
            train_image.append(imgsAll[i])
            train_label.append(labelBinary[i])
            train_label_orig.append(labelOrigin[i])

        for i in range(cut + 1, len(labelsString)):
            test_image.append(imgsAll[i])
            test_label.append(labelBinary[i])

        #for k mean
        #test_image=train_image
        #train_label = train_label_orig
        #test_label=train_label_orig

    elif dataset=='MNIST_ORIG_EVEN_ODD' or dataset=='MNIST_ORIG_ALL_LABELS':
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


    elif dataset=='EMNIST':

        from datasets.emnist_extractor import read_data
        clients, groups, train_data, test_client_to_data_dict = read_data(dataset_file_path +'/femnist/train',dataset_file_path +'/femnist/test' )
        #print('clients',clients)
        #print('groups',groups)


        list_train_keys=list(train_data.keys())
        #print('list train keys',len(list_train_keys))
        train_image=[]
        train_label=[]
        train_label_orig=[]



        for i in range(0,len(list_train_keys)):
            # note: each time we append a list
            train_image+= train_data[list_train_keys[i]]["x"]
            train_label+= train_data[list_train_keys[i]]["y"]
            #print('client',i)
            #print(len(train_data[list_train_keys[i]]["y"]))
            for j in range(0, len(train_data[list_train_keys[i]]["x"])):
                train_label_orig.append(i)

        #print('train orig',train_label_orig)
        #print('len',len(train_label_orig))

        #print('training orgin', len(train_label_orig))
        #print('train image shape')
        #print(np.array(train_image).shape)
        for i in range(0,len(train_label)):
            train_label[i]=get_one_hot_from_label_index(train_label[i],62)

        test_image = []
        test_label = []
        test_label_orig=[]

        list_test_keys=list(test_client_to_data_dict.keys())
        #print('test')
        #print(list(test_data.keys()))
        #print('len test',len(list(test_data.keys())))
        for i in range(0,len(list_test_keys)):
            test_image+= test_client_to_data_dict[list_test_keys[i]]["x"]
            test_label+= test_client_to_data_dict[list_test_keys[i]]["y"]
            for j in range(0, len(test_client_to_data_dict[list_test_keys[i]]["x"])):
                test_label_orig.append(i)

        #print('test orig', test_label_orig)
        #print('len test orig', len(test_label_orig))
        for i in range(0,len(test_label)):
            test_label[i]=get_one_hot_from_label_index(test_label[i],62)

        #print('test shape')
        #print(np.array(test_image).shape)
        #print(np.array(test_label).shape)

        #print('test label or', len(test_label))
        #print('test label orgine',test_label_orig)


    elif dataset == 'EMNIST_DEFAULT':

        from datasets.emnist_extractor import read_data
        clients, groups, train_data, test_client_to_data_dict = read_data(dataset_file_path + '/femnist-default/train',
                                                                          dataset_file_path + '/femnist-default/test')
        #print('clients',clients)
        #print(len(clients))

        # print('groups',groups)

        list_train_keys = list(train_data.keys())
        # print('list train keys',len(list_train_keys))
        train_image = []
        train_label = []
        train_label_orig = []

        for i in range(0, len(list_train_keys)):
            # note: each time we append a list
            train_image += train_data[list_train_keys[i]]["x"]
            train_label += train_data[list_train_keys[i]]["y"]
            # print('client',i)
            # print(len(train_data[list_train_keys[i]]["y"]))
            for j in range(0, len(train_data[list_train_keys[i]]["x"])):
                train_label_orig.append(i)

        #print('train orig', train_label_orig)
        #print('len', len(train_label_orig))

        # print('training orgin', len(train_label_orig))
        # print('train image shape')
        # print(np.array(train_image).shape)
        for i in range(0, len(train_label)):
            train_label[i] = get_one_hot_from_label_index(train_label[i], 62)

        test_image = []
        test_label = []
        test_label_orig = []

        list_test_keys = list(test_client_to_data_dict.keys())
        # print('test')
        # print(list(test_data.keys()))
        # print('len test',len(list(test_data.keys())))
        for i in range(0, len(list_test_keys)):
            test_image += test_client_to_data_dict[list_test_keys[i]]["x"]
            test_label += test_client_to_data_dict[list_test_keys[i]]["y"]
            for j in range(0, len(test_client_to_data_dict[list_test_keys[i]]["x"])):
                test_label_orig.append(i)

        #print('test orig', test_label_orig)
        #print('len test orig', len(test_label_orig))
        for i in range(0, len(test_label)):
            test_label[i] = get_one_hot_from_label_index(test_label[i], 62)

        # print('test shape')
        # print(np.array(test_image).shape)
        # print(np.array(test_label).shape)

        # print('test label or', len(test_label))
        # print('test label orgine',test_label_orig)




    elif dataset=='EMNIST30':

        from datasets.emnist_extractor import read_data
        clients, groups, train_data, test_client_to_data_dict = read_data(dataset_file_path +'/femnist30/train',dataset_file_path +'/femnist30/test' )
        #print('clients',clients)
        #print('groups',groups)

        list_train_keys=list(train_data.keys())
        #print('list train keys',len(list_train_keys))
        train_image=[]
        train_label=[]
        train_label_orig=[]



        for i in range(0,len(list_train_keys)):
            # note: each time we append a list
            train_image+= train_data[list_train_keys[i]]["x"]
            train_label+= train_data[list_train_keys[i]]["y"]
            #print('attention',len(train_data[list_train_keys[i]]["y"]))

            #print('client',i)
            #print(len(train_data[list_train_keys[i]]["y"]))
            for j in range(0, len(train_data[list_train_keys[i]]["x"])):
                train_label_orig.append(i)

        #print('train orig',train_label_orig)
        #print('training orgin', len(train_label_orig))
        #print('train image shape')
        #print(np.array(train_image).shape)
        for i in range(0,len(train_label)):
            train_label[i]=get_one_hot_from_label_index(train_label[i],62)

        test_image = []
        test_label = []
        test_label_orig=[]

        list_test_keys=list(test_client_to_data_dict.keys())
        #print('test')
        #print(list(test_data.keys()))
        #print('len test',len(list(test_data.keys())))
        for i in range(0,len(list_test_keys)):
            test_image+= test_client_to_data_dict[list_test_keys[i]]["x"]
            test_label+= test_client_to_data_dict[list_test_keys[i]]["y"]
            for j in range(0, len(test_client_to_data_dict[list_test_keys[i]]["x"])):
                test_label_orig.append(i)


        for i in range(0,len(test_label)):
            test_label[i]=get_one_hot_from_label_index(test_label[i],62)

        #print('test shape')
        #print(np.array(test_image).shape)
        #print(np.array(test_label).shape)

        #print('test label or', len(test_label))
        #print('test label orgine',test_label_orig)


    elif dataset=='PRINT_62':
        train_image=np.load(dataset_file_path+'/printed-digits/train-image.npy')
        train_label=np.load(dataset_file_path+'/printed-digits/train-label.npy')
        test_image=np.load(dataset_file_path+'/printed-digits/test-image.npy')
        test_label=np.load(dataset_file_path+'/printed-digits/test-label.npy')
        train_label_orig=[]

    elif dataset == 'CHAR_74':
        train_image = np.load(dataset_file_path + '/Char74/train-image.npy')
        train_label = np.load(dataset_file_path + '/Char74/train-label.npy')
        test_image = np.load(dataset_file_path + '/Char74/test-image.npy')
        test_label = np.load(dataset_file_path + '/Char74/test-label.npy')
        train_label_orig = []

    elif dataset == 'SVNH':
        train_image = np.load(dataset_file_path + '/SVNH/train_image_svnh.npy')
        train_label = np.load(dataset_file_path + '/SVNH/train_label_svnh_hot.npy')
        test_image = np.load(dataset_file_path + '/SVNH/test_image_svnh.npy')
        test_label = np.load(dataset_file_path + '/SVNH/test_label_svnh_hot.npy')
        train_label_orig = []

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

    elif dataset == 'CIFAR_100':
        train_image = np.load(dataset_file_path + '/cifar-100/train_image.npy')
        train_label = np.load(dataset_file_path + '/cifar-100/train_label.npy')
        test_image = np.load(dataset_file_path + '/cifar-100/test_image.npy')
        test_label = np.load(dataset_file_path + '/cifar-100/test_label.npy')
        train_label_orig = np.load(dataset_file_path + '/cifar-100/train_label_orig.npy')


    elif dataset == 'CIFAR_100_GRAY':
        train_image = np.load(dataset_file_path + '/cifar-100-gray-28by28/cifar100_train_image_gray.npy')
        train_label = np.load(dataset_file_path + '/cifar-100-gray-28by28/cifar100_train_label_gray.npy')
        test_image = np.load(dataset_file_path + '/cifar-100-gray-28by28/cifar100_test_image_gray.npy')
        test_label = np.load(dataset_file_path + '/cifar-100-gray-28by28/cifar100_test_label_gray.npy')
        train_label_orig = np.load(dataset_file_path + '/cifar-100-gray-28by28/train_label_orig.npy')


    elif dataset == 'CIFAR_100_GRAY_62':
        train_image = np.load(dataset_file_path + '/cifar-100-gray-28by28-62classes/train_image.npy')
        train_label = np.load(dataset_file_path + '/cifar-100-gray-28by28-62classes/train_label.npy')
        test_image = np.load(dataset_file_path + '/cifar-100-gray-28by28-62classes/test_image.npy')
        test_label = np.load(dataset_file_path + '/cifar-100-gray-28by28-62classes/test_label.npy')
        train_label_orig = np.load(dataset_file_path + '/cifar-100-gray-28by28-62classes/train_label_orig.npy')



    elif dataset=='CIFAR_10':
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

        train_label_orig=[]
        for i in range(0, len(train_label)):
            label = get_index_from_one_hot_label(train_label[i])
            train_label_orig.append(label[0])


    elif dataset=='SIMPLE_REG':

        X = np.random.uniform(-1.0, 1.0, 600)

        # Assume 5 different original labels
        yorig = np.random.random_integers(0, 4, 600)

        y_coeff = yorig / 10.0 + 1.0

        y = np.inner(X, y_coeff) + np.random.uniform(-0.005, 0.005, 600)

        cut = 600
        # cut = 300

        train_image = []
        test_image = []
        train_label = []
        test_label = []
        train_label_orig = []
        for i in range(0, cut):
            train_image.append([X[i]])
            train_label.append(y[i])
            train_label_orig.append(yorig[i])

        for i in range(cut + 1, len(yorig)):
            test_image.append([X[i]])
            test_label.append(y[i])

    elif dataset=='FACEBOOK':
        from xlrd import open_workbook

        wb = open_workbook(dataset_file_path + '/facebook_data.xls')

        X = np.zeros((495, 19))

        for s in wb.sheets():

            for j in range(0, 495):
                for i in range(0, 19):
                    X[j][i] = (s.cell(j, i).value)

        # y_orig is used to determine the data to node assignment in different cases

        # y_orig = []
        # for j in range(0, 495):
        #     y_orig.append(int(X[j][1]))

        normalize_data(X)

        from sklearn.cluster import SpectralClustering
        spectral = SpectralClustering()
        y_orig = spectral.fit_predict(X)

        y = []

        for j in range(0, 495):
            y.append(X[j][18])

        X = X[0:495, 0:18]

        # TODO: This may not need a separate test data

        # cut = 250
        cut = 495

        train_image = []
        test_image = []
        train_label = []
        test_label = []
        train_label_orig = []
        for i in range(0, cut):
            train_image.append(X[i])
            train_label.append(y[i])
            train_label_orig.append(y_orig[i])

        for i in range(cut + 1, len(y_orig)):
            test_image.append(X[i])
            test_label.append(y[i])


    elif dataset=='ENERGY':
        from xlrd import open_workbook

        wb = open_workbook(dataset_file_path + '/energydata_complete.xlsx')

        X = np.zeros((19735, 28))

        for s in wb.sheets():

            for j in range(0, 19735):
                for i in range(0, 28):
                    X[j][i] = s.cell(j + 1, i + 1).value   # First row in dataset is header, thus j + 1;
                                                           # First column in dataset is date, this i + 1

        # # y_orig is used to determine the data to node assignment in different cases
        #
        # y_orig = []
        # for j in range(0, 19735):
        #     y_orig.append(int(X[j][1] / 10))

        normalize_data(X)

        from sklearn.cluster import MiniBatchKMeans
        cluster = MiniBatchKMeans(n_clusters=20)
        y_orig = cluster.fit_predict(X)    # y_orig is used to determine the data to node assignment in different cases
 

        y = []

        for j in range(0, 19735):
            y.append(X[j][0])

        X = X[0:19735, 2:28]

        # from sklearn.cluster import MiniBatchKMeans
        # cluster = MiniBatchKMeans(n_clusters=10)
        # y_orig = cluster.fit_predict(np.reshape(y, (-1,1)))    # y_orig is used to determine the data to node assignment in different cases
        # np.set_printoptions(threshold=np.nan)
        # print(y_orig)


        # TODO: This may not need a separate test data

        cut = 19735

        train_image = []
        test_image = []
        train_label = []
        test_label = []
        train_label_orig = []
        for i in range(0, cut):
            train_image.append(X[i])
            train_label.append(y[i])
            train_label_orig.append(y_orig[i])

        for i in range(cut + 1, len(y_orig)):
            test_image.append(X[i])
            test_label.append(y[i])

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

    elif dataset=='CIFAR_10':
        from datasets.cifar_10_extractor import cifar_10_extract_samples

        train_image, train_label = cifar_10_extract_samples(samples_list, True, dataset_file_path)

    elif dataset=='USER_MODEL':
        from datasets.user_model_extractor import user_model_extract_samples

        train_image,train_label = user_model_extract_samples(samples_list, dataset_file_path)

    else:
        raise Exception('Training data sampling not supported for the given dataset name, use entire dataset by setting batch_size = total_data, ' +
                        'also confirm that dataset name is correct.')


    return train_image,train_label
