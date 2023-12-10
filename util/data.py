from datasets.dataset import get_index_from_one_hot_label
import math
import numpy as np
from scipy import stats
import random

def reverse_mnist(train_image_mnist):
    train_image_mnist_reverse = []
    for i in range(0, len(train_image_mnist)):
        train_image_mnist_reverse.append([])

    for i in range(0, len(train_image_mnist)):
        train_image_mnist_reverse[i] = 1 - train_image_mnist[i] *np.random.randint(5)

    return train_image_mnist_reverse


def get_lists_indices_per_label(n_category,validation_label):
    list_indices_validation = []


    for i in range(0, n_category):
        list_indices_validation.append([])
    for i in range(0, n_category):
        for j in range(0, len(validation_label)):
            if get_index_from_one_hot_label(validation_label[j])[0] == i:
                list_indices_validation[i].append(j)

    return list_indices_validation


def get_label_missing(n_category,lists_indices_per_label):

    no_label = []
    for i in range(0, n_category):
        if len(lists_indices_per_label[i]) == 0:
            no_label.append(i)
    return no_label


def get_training_testing_partition( n_category, list_indices_validation, validation_image, validation_label):

    validation_training = []
    validation_testing = []
    for i in range(0, n_category):
        for j in range(0, math.ceil(len(list_indices_validation[i]) / 1.3)):
            validation_training.append(list_indices_validation[i][j])
    #validation training and validation testing are arrazy of indices
    validation_testing = [i for i in range(0, len(validation_label)) if i not in validation_training]

    validation_training_image = [validation_image[i] for i in range(0, len(validation_image)) if i in validation_training]
    validation_training_label = [validation_label[i] for i in range(0, len(validation_image)) if i in validation_training]

    validation_testing_image = [validation_image[i] for i in range(0, len(validation_image)) if
                                i in validation_testing]
    validation_testing_label = [validation_label[i] for i in range(0, len(validation_image)) if
                                i in validation_testing]

    return validation_training,validation_testing,validation_training_image,validation_training_label,validation_testing_image,validation_testing_label


def adapt_test_to_validation(label_missing_in_validation,list_indices_test,test_image_no_validation,test_label_no_validation):

    all_indices_to_remove_from_test = []
    # print('no label',no_label)

    for i in label_missing_in_validation:

        all_indices_to_remove_from_test.extend(list_indices_test[i])

    test_image_no_validation_clean = [test_image_no_validation[i] for i in range(0, len(test_image_no_validation)) if
                                      i not in all_indices_to_remove_from_test]
    test_label_no_validation_clean = [test_label_no_validation[i] for i in range(0, len(test_image_no_validation)) if
                                      i not in all_indices_to_remove_from_test]

    return test_image_no_validation_clean, test_label_no_validation_clean

def write_dictionary_validation(results_file_path,subfilename,dic_ind_grad_validation, validation_testing_indices):

    with open(results_file_path + subfilename, 'a') as f:
        for s in validation_testing_indices:
            f.write(str(dic_ind_grad_validation[s]) + '\n')
        f.close()


def write_dictionary_data(results_file_path,subfilename,sorted_d):
    with open(results_file_path + str(subfilename), 'a') as f:

        for i in range(0, len(sorted_d)):
            f.write(str(sorted_d[i]) + '\n')
        f.close()


def get_n_sample_to_keep_from_save_file(results_file_path,total_n_sample): # select n_sample to keep

    distance_list_validation = []
    with open(str(results_file_path) + '/dico-validation.csv') as f:
        for line in f:
            # l = line.replace('\n','').split(',')
            l = line.replace('[', '')
            l = l.replace(']', '')
            l = l.replace('(', '')
            l = l.replace(')', '')
            l = l.split(',')
            distance_list_validation.append(float(l[0]))

    indices_list = []
    distance_list = []
    node_list = []
    label_list = []
    with open(str(results_file_path) + '/dico-order.csv') as f:
        for line in f:
            # l = line.replace('\n','').split(',')
            l = line.replace('[', '')
            l = l.replace(']', '')
            l = l.replace('(', '')
            l = l.replace(')', '')
            l = l.split(',')
            indices_list.append(int(l[0]))
            distance_list.append(float(l[1]))
            node_list.append(int(l[2]))
            label_list.append(int(l[3]))

    max_each_cut = []
    #steps = np.arange(10, total_n_sample, 100)
    steps = np.arange(10, total_n_sample, 1000)

    list_cut = []
    for i in steps:
        list_cut.append(distance_list[i])

    cdf = stats.cumfreq(distance_list_validation, numbins=100, defaultreallimits=(0, 30))  # 150

    for t in steps:
        cdf2 = stats.cumfreq(distance_list[0:t], numbins=100, defaultreallimits=(0, 30))  # 150

        difference = abs(cdf.cumcount / cdf.cumcount[-1] - cdf2.cumcount / cdf2.cumcount[-1])

        max_value = max(difference)

        max_each_cut.append(max_value)



    n_sample_to_keep=steps[np.argmin(max_each_cut)]

    indices_to_keep = []
    for i in range(0, n_sample_to_keep):
        # print(i)
        #indices_to_keep.append(distance_list[i])
        indices_to_keep.append(indices_list[i])
    return n_sample_to_keep,indices_to_keep

def get_n_sample_to_keep(results_file_path,total_n_sample,sorted_d): # select n_sample to keep
    print('sorted,',sorted_d)
    distance_list_validation = []
    with open(str(results_file_path) + '/dico-validation.csv') as f:
        for line in f:
            # l = line.replace('\n','').split(',')
            l = line.replace('[', '')
            l = l.replace(']', '')
            l = l.replace('(', '')
            l = l.replace(')', '')
            l = l.split(',')
            distance_list_validation.append(float(l[0]))

    indices_list = []
    distance_list = []
    node_list = []
    label_list = []
    with open(str(results_file_path) + '/dico-order.csv') as f:
        for line in f:
            # l = line.replace('\n','').split(',')
            l = line.replace('[', '')
            l = l.replace(']', '')
            l = l.replace('(', '')
            l = l.replace(')', '')
            l = l.split(',')
            indices_list.append(int(l[0]))
            distance_list.append(float(l[1]))
            node_list.append(int(l[2]))
            label_list.append(int(l[3]))

    max_each_cut = []
    steps = np.arange(10, len(distance_list), 100)


    list_cut = []
    for i in steps:
        list_cut.append(distance_list[i])

    cdf = stats.cumfreq(distance_list_validation, numbins=100, defaultreallimits=(0, 30))  # 150

    for t in steps:
        cdf2 = stats.cumfreq(distance_list[0:t], numbins=100, defaultreallimits=(0, 30))  # 150

        difference = abs(cdf.cumcount / cdf.cumcount[-1] - cdf2.cumcount / cdf2.cumcount[-1])

        max_value = max(difference)

        max_each_cut.append(max_value)



    n_sample_to_keep=steps[np.argmin(max_each_cut)]
    lambda_star = distance_list[n_sample_to_keep]
    print('lambda star', lambda_star)
    print('p', n_sample_to_keep/total_n_sample)
    print('n sample to keep',n_sample_to_keep)

    indices_to_keep = []
    for i in range(0, n_sample_to_keep):
        # print(i)

        indices_to_keep.append(sorted_d[i][0])
        # indices_to_keep.append(indices_list[i])
    return n_sample_to_keep,indices_to_keep


def select_intersection(indices_to_keep,indices_each_node_case_copy_init):
    v = np.sort(indices_to_keep)
    t = np.sort(indices_each_node_case_copy_init)
    #print('indices to keep',v)
    #print('all',t)

    keep = []
    j = 0
    for i in v:
        if t[j] > i:
            continue
        else:
            while t[j] < i:
                j += 1
                if j>len(t)-1:
                    return keep
            if t[j] == i:
                keep.append(i)
    #print('intersection',keep)
    return keep

def get_train_label_orig_from_hot(train_label):
    train_label_orig=[]
    for i in train_label:
        train_label_orig.append(get_index_from_one_hot_label(i)[0])

    return train_label_orig

def get_assignment_index_each_node(train_label_orig,indices_shift,number_element_per_node_this_dataset,n_nodes,counter_dataset):
    d = {}
    maxCase=2
    indexesEachNodeCase = []
    for i in range(0, maxCase):
        indexesEachNodeCase.append([])

    for i in range(0, n_nodes):
        for j in range(0, maxCase):
            indexesEachNodeCase[j].append([])

    #randomly shift indices for case0
    list= np.arange(0,len(train_label_orig))
    random.shuffle(list)
    train_label_orig=[train_label_orig[i] for i in list]
    indices_shift=[indices_shift[i] for i in list]

    # case 0
    for i in range(0,n_nodes):

        j=0
        while len(indexesEachNodeCase[0][i])<(number_element_per_node_this_dataset[i]):
            indexesEachNodeCase[0][i].append(indices_shift[j])
            j=j+1



    #### case 1

    unique_label=np.unique(train_label_orig)
    print('unique label',unique_label)
    indices_by_label=[]
    for i in range(0,len(unique_label)):
        indices_by_label.append([])
    for j in range(0, len(unique_label)):
        for i in range(0,len(train_label_orig)):
            if train_label_orig[i]==j:
                indices_by_label[j].append(indices_shift[i])

    #print('indices by label',len(indices_by_label))
    #print(indices_by_label[61])



    if len(unique_label)==n_nodes:
        for i in range(0,n_nodes):
            indexesEachNodeCase[1][i].extend(indices_by_label[i])


    elif n_nodes>len(unique_label):
        shift=counter_dataset
        # mapping nodes into labels

        for i in range(0,len(unique_label)):
            d[i]=[]

        for i in range(0,n_nodes):
            j=i+shift
            d[j%len(unique_label)].append(i)


        for i in range(0,len(unique_label)):

            #tell us how many instance each node  (that was assigned this label) should have of this label
            number_element_per_node_this_label = get_assigment_node(indices_by_label[i], len(d[i]))
            #print('number',number_element_per_node_this_label)
            #print('d[i]',d[i])
            for n in range(0,len(d[i])):

                #print(d[i])

                j=0

                while len(indexesEachNodeCase[1][d[i][n]]) < (number_element_per_node_this_label[n]):

                    indexesEachNodeCase[1][d[i][n]].append(indices_by_label[i][j])
                    j = j + 1
                    #print('j',j)




            d[i].append(number_element_per_node_this_label)

        #print('dico',d)

    return indexesEachNodeCase,d

def get_assigment_node(train_label_orig,n_nodes):

    # function that given  the len of an arary and number of nodes, find the number of elments to put in each node (not balanced)


    # first we want to know the number of elemenent per nodes

    len_dataset=len(train_label_orig)
    len_black_interval=int(len_dataset/n_nodes)

    interval_black_interm = np.arange(0,len_dataset, len_black_interval)

    if len(interval_black_interm)<n_nodes+1:
        interval_black=(interval_black_interm.tolist())
        interval_black.append(len_dataset)
    else:
        interval_black = (interval_black_interm.tolist())


    blue_internval=[]
    for i in range(0,len(interval_black)-1):
        blue_internval.append(int((interval_black[i]+interval_black[i+1])/2))


    # select number of nodes
    x_i=[]
    #borne inferieur
    x_i.append(0)
    for i in range(0,len(blue_internval)-1):
        x_i.append(random.randint(blue_internval[i],blue_internval[i+1]))
    # borne superieur
    x_i.append(len_dataset)


    element_per_node=[]
    for j in range(0,len(x_i)-1):
        element_per_node.append(x_i[j+1]-x_i[j])


    return element_per_node

def get_assignment_index_each_node(cfg,train_label_orig,indices_shift,number_element_per_node_this_dataset,n_nodes,counter_dataset):
    d = {}
    maxCase=cfg.maxCase
    indexesEachNodeCase = []
    for i in range(0, maxCase):
        indexesEachNodeCase.append([])

    for i in range(0, n_nodes):
        for j in range(0, maxCase):
            indexesEachNodeCase[j].append([])

    #randomly shift indices for case0
    list= np.arange(0,len(train_label_orig))
    random.shuffle(list)
    train_label_orig=[train_label_orig[i] for i in list]
    indices_shift=[indices_shift[i] for i in list]

    # case 0
    for i in range(0,n_nodes):

        j=0
        while len(indexesEachNodeCase[0][i])<(number_element_per_node_this_dataset[i]):
            indexesEachNodeCase[0][i].append(indices_shift[j])
            j=j+1



    #### case 1

    unique_label=np.unique(train_label_orig)

    indices_by_label=[]
    for i in range(0,len(unique_label)):
        indices_by_label.append([])
    for j in range(0, len(unique_label)):
        for i in range(0,len(train_label_orig)):
            if train_label_orig[i]==j:
                indices_by_label[j].append(indices_shift[i])




    if len(unique_label)==n_nodes:
        for i in range(0,n_nodes):
            indexesEachNodeCase[1][i].extend(indices_by_label[i])


    elif n_nodes>len(unique_label):
        shift=counter_dataset
        # mapping nodes into labels

        for i in range(0,len(unique_label)):
            d[i]=[]

        for i in range(0,n_nodes):
            j=i+shift
            d[j%len(unique_label)].append(i)


        for i in range(0,len(unique_label)):

            #tell us how many instance each node  (that was assigned this label) should have of this label
            number_element_per_node_this_label = get_assigment_node(indices_by_label[i], len(d[i]))
            #print('number',number_element_per_node_this_label)
            #print('d[i]',d[i])
            for n in range(0,len(d[i])):

                #print(d[i])

                j=0

                while len(indexesEachNodeCase[1][d[i][n]]) < (number_element_per_node_this_label[n]):

                    indexesEachNodeCase[1][d[i][n]].append(indices_by_label[i][j])
                    j = j + 1
                    #print('j',j)




            d[i].append(number_element_per_node_this_label)

        #print('dico',d)

    return indexesEachNodeCase,d

# def get_assignment_index_each_node(train_label_orig,indices_shift,number_element_per_node_this_dataset,n_nodes,counter_dataset):
#     d = {}
#     maxCase=4
#     indexesEachNodeCase = []
#     for i in range(0, maxCase):
#         indexesEachNodeCase.append([])
#
#     for i in range(0, n_nodes):
#         for j in range(0, maxCase):
#             indexesEachNodeCase[j].append([])
#
#     #randomly shift indices for case0
#     list= np.arange(0,len(train_label_orig))
#     random.shuffle(list)
#     train_label_orig=[train_label_orig[i] for i in list]
#     indices_shift=[indices_shift[i] for i in list]
#
#     # case 0
#     for i in range(0,n_nodes):
#
#         j=0
#         while len(indexesEachNodeCase[0][i])<(number_element_per_node_this_dataset[i]):
#             indexesEachNodeCase[0][i].append(indices_shift[j])
#             j=j+1
#
#
#
#     #### case 1
#
#     unique_label=np.unique(train_label_orig)
#     #print('unique label',unique_label)
#     indices_by_label=[]
#     for i in range(0,len(unique_label)):
#         indices_by_label.append([])
#     for j in range(0, len(unique_label)):
#         for i in range(0,len(train_label_orig)):
#             if train_label_orig[i]==j:
#                 indices_by_label[j].append(indices_shift[i])
#
#     #print('indices by label',len(indices_by_label))
#     #print(indices_by_label[61])
#
#
#
#     if len(unique_label)==n_nodes:
#         for i in range(0,n_nodes):
#             indexesEachNodeCase[1][i].extend(indices_by_label[i])
#
#
#     elif n_nodes>len(unique_label):
#         shift=counter_dataset
#         # mapping nodes into labels
#
#         for i in range(0,len(unique_label)):
#             d[i]=[]
#
#         for i in range(0,n_nodes):
#             j=i+shift
#             d[j%len(unique_label)].append(i)
#
#
#         for i in range(0,len(unique_label)):
#
#             #tell us how many instance each node  (that was assigned this label) should have of this label
#             number_element_per_node_this_label = get_assigment_node(indices_by_label[i], len(d[i]))
#             #print('number',number_element_per_node_this_label)
#             #print('d[i]',d[i])
#             for n in range(0,len(d[i])):
#
#                 #print(d[i])
#
#                 j=0
#
#                 while len(indexesEachNodeCase[1][d[i][n]]) < (number_element_per_node_this_label[n]):
#
#                     indexesEachNodeCase[1][d[i][n]].append(indices_by_label[i][j])
#                     j = j + 1
#                     #print('j',j)
#
#
#
#
#             d[i].append(number_element_per_node_this_label)
#
#         #print('dico',d)
#
#     return indexesEachNodeCase,d

# def get_indices_each_node_case2(n_nodes, maxCase, label_list):
#     indexesEachNodeCase = []
#     for i in range(0, maxCase):
#         indexesEachNodeCase.append([])
#
#     for i in range(0, n_nodes):
#         for j in range(0, maxCase):
#             indexesEachNodeCase[j].append([])
#
#     # indexesEachNode is a big list that contains N-number of sublists. Sublist n contains the indexes that should be assigned to node n
#
#     minLabel = min(label_list)
#     maxLabel = max(label_list)
#     numLabels = maxLabel - minLabel + 1
#
#     for i in range(0, len(label_list)):
#
#         # case 1
#
#         #randomperm = np.random.permutation(len(train))
#         #indexesEachNodeCase[0][(i % n_nodes)].append(randomperm[i])
#         indexesEachNodeCase[0][(i % n_nodes)].append(i)
#         # if i % n_nodes==0:
#         #     print('-----')
#         #     print('label',label_list[i])
#         #     print('indice',i)
#         #     print('-----')
#
#         # case 2
#
#         tmp_target_node=int((label_list[i]-minLabel) %n_nodes)
#         if n_nodes>numLabels:
#             tmpMinIndex=0
#             tmpMinVal=math.inf
#             for n in range(0,n_nodes):
#                 if (n)%numLabels==tmp_target_node and len(indexesEachNodeCase[1][n])<tmpMinVal:
#                     tmpMinVal=len(indexesEachNodeCase[1][n])
#                     tmpMinIndex=n
#             tmp_target_node=tmpMinIndex
#
#         indexesEachNodeCase[1][tmp_target_node].append(i)
#
#
#     return indexesEachNodeCase



def get_index_from_one_hot_label(label):
    for i in range(0, len(label)):
        if (label[i] == 1):
            return [i]



def get_even_odd_from_one_hot_label(label):
    for i in range(0, len(label)):
        if (label[i] == 1):
            c=i%2
            if c==0:
                c=1
            elif c==1:
                c=-1
            return c


def get_one_hot_from_label_index(label, number_of_labels=10):
    one_hot = np.zeros(number_of_labels)
    one_hot[label] = 1
    return one_hot


def normalize_data(X):
    for i in range(0, len(X[0,:])):
       X[:,i] = (X[:,i]-np.mean(X[:,i]))/np.std(X[:,i])
