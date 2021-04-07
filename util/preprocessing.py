from util.data import select_intersection, get_train_label_orig_from_hot, get_assigment_node, get_assignment_index_each_node, get_lists_indices_per_label, get_label_missing, write_dictionary_data, write_dictionary_validation, get_n_sample_to_keep, get_training_testing_partition,get_index_from_one_hot_label
import copy
import numpy as np
import math
from datasets.dataset import get_data
import operator

train_image_all = []
train_label_all = []
test_image_all = []
test_label_all = []
train_label_orig_all = []
keep_track_indices_dataset_train = []
keep_track_indices_dataset_test = []


def multiple_to_single_dataset(cfg):
    cfg.case = 1
    cfg.maxCase = 2

    counter_dataset = 0

    indices_each_node_case = []
    for i in range(0, cfg.maxCase):
        indices_each_node_case.append([])

    for i in range(0, cfg.n_nodes):
        for j in range(0, cfg.maxCase):
            indices_each_node_case[j].append([])


    for dataset in cfg.dataset_list:

        indices_init_train = len(train_label_all)
        indices_init_test = len(test_label_all)


        train_image, train_label, test_image, test_label, train_label_orig = get_data(
            dataset,
            cfg.total_data,
            cfg.dataset_file_path,
            use_test_client_to_data_dict=False)

        # make sure every dataset has train_label_orig
        if len(train_label_orig) == 0:
            train_label_orig = get_train_label_orig_from_hot(train_label)
        train_image_all.extend(train_image)
        train_label_all.extend(train_label)
        test_image_all.extend(test_image)
        test_label_all.extend(test_label)



        number_element_per_node_this_dataset = get_assigment_node(
            train_label_orig, cfg.n_nodes)

        assignment_index_each_node, d = get_assignment_index_each_node(cfg,
            train_label_orig,
            np.arange(indices_init_train, len(train_label_all)),
            number_element_per_node_this_dataset, cfg.n_nodes,
            counter_dataset)
        counter_dataset += 1
        np.save(
            cfg.results_file_path + '/d-' + str(counter_dataset) + '.npy',
            d)



        for c in range(0, cfg.maxCase):
            for i in range(0, cfg.n_nodes):
                indices_each_node_case[c][i].extend(
                    assignment_index_each_node[c][i])

        # indices_each_node.extend()

        ############################################################################################

        keep_track_indices_dataset_train.append(
            np.arange(indices_init_train, len(train_label_all)))
        keep_track_indices_dataset_test.append(
            np.arange(indices_init_test, len(test_label_all)))

    return train_image_all, train_label_all, keep_track_indices_dataset_train, keep_track_indices_dataset_test, indices_each_node_case


def split_focus_noise(keep_track_indices_dataset_train,
                      keep_track_indices_dataset_test):
    # useful for the end to compute loss and accurady
    train_image_focus = [
        train_image_all[i] for i in keep_track_indices_dataset_train[0]
    ]
    train_label_focus = [
        train_label_all[i] for i in keep_track_indices_dataset_train[0]
    ]
    test_image_focus = [
        test_image_all[i] for i in keep_track_indices_dataset_test[0]
    ]
    test_label_focus = [
        test_label_all[i] for i in keep_track_indices_dataset_test[0]
    ]

    train_image_noise = [
        train_image_all[i] for i in range(0, len(train_label_all))
        if i not in keep_track_indices_dataset_train[0]
    ]
    train_label_noise = [
        train_label_all[i] for i in range(0, len(train_label_all))
        if i not in keep_track_indices_dataset_train[0]
    ]
    test_image_noise = [
        test_image_all[i] for i in range(0, len(test_label_all))
        if i not in keep_track_indices_dataset_test[0]
    ]
    test_label_noise = [
        test_label_all[i] for i in range(0, len(test_label_all))
        if i not in keep_track_indices_dataset_test[0]
    ]

    return train_image_focus, train_label_focus, test_image_focus, test_label_focus, train_image_noise, train_label_noise, test_image_noise, test_label_noise


# only on the focus dataset


def train_validation(indices_each_node_case_copy_init, model, dim_w, sim, cfg,
                     lists_indices_per_label_validation, validation_image,
                     validation_label):
    # split benchmark dataset into training and testing
    validation_training_indices, validation_testing_indices, validation_training_image, validation_training_label, validation_testing_image, validation_testing_label = get_training_testing_partition(
        cfg.n_category, lists_indices_per_label_validation, validation_image,
        validation_label)

    dic_ind_grad = {}
    dic_ind_grad_validation = {}

    #print('-----Start of the validation training----------------')
    model.start_consecutive_training(
        model.get_init_weight(dim_w, rand_seed=sim))

    max_val = math.inf
    counter_convergence_reached = 0
    i = 0
    while counter_convergence_reached < 200:

        model.run_one_step_consecutive_training(validation_image,
                                                validation_label,
                                                validation_training_indices)
        w_validation = model.end_consecutive_training_and_get_weights()

        loss_train = model.loss(validation_training_image,
                                validation_training_label, w_validation)
        loss_test = model.loss(validation_testing_image,
                               validation_testing_label, w_validation)
        accu_train = model.accuracy(validation_training_image,
                                    validation_training_label, w_validation)
        accu_test = model.accuracy(validation_testing_image,
                                   validation_testing_label, w_validation)

        if loss_test < max_val:
            max_val = loss_test
            counter_convergence_reached = 0
        else:
            if i > 10:
                counter_convergence_reached += 1
                #print('counter', counter_convergence_reached)

        i = i + 1
        if i > 10000:
            break

        #print('accu train', model.accuracy(validation_training_image, validation_training_label, w_validation))
        #print('accu test', model.accuracy(validation_testing_image, validation_testing_label, w_validation))

        with open(cfg.results_file_path + '/loss-accu-validation.csv',
                  'a') as f:
            f.write(
                str(i) + ',' + str(loss_train) + ',' + str(loss_test) + ',' +
                str(accu_train) + ',' + str(accu_test) + '\n')
            f.close()

    for s in validation_testing_indices:
        norm_diff = model.loss(validation_image[s:s + 1],
                               validation_label[s:s + 1], w_validation)
        dic_ind_grad_validation[s] = (norm_diff, None,
                                      get_index_from_one_hot_label(
                                          validation_label[s]))

    write_dictionary_validation(cfg.results_file_path, '/dico-validation.csv',
                                dic_ind_grad_validation,
                                validation_testing_indices)

    # RANK ALL DATA POINT IN EACH NODE
    for n in range(0, cfg.n_nodes):

        norm_diff_vec = model.loss_vector(train_image_all,train_label_all,w_validation,indices_each_node_case_copy_init[cfg.case][n])
        #print(len(norm_diff_vec))
        #print('check',len(indices_each_node_case_copy_init[cfg.case][n]))
        count = 0

        for s in indices_each_node_case_copy_init[cfg.case][n]:

            #norm_diff = model.loss(train_image_all[s:s + 1],
                                  # train_label_all[s:s + 1], w_validation)

            dic_ind_grad[s] = (norm_diff_vec[count], n,
                               get_index_from_one_hot_label(
                                   train_label_all[s]))
            count+=1

    sorted_d = sorted(dic_ind_grad.items(), key=operator.itemgetter(1))
    #print('here',sorted_d)

    write_dictionary_data(cfg.results_file_path, '/dico-order.csv', sorted_d)

    n_sample_keep, indices_to_keep = get_n_sample_to_keep(
        cfg.results_file_path, len(train_label_all), sorted_d)
    # print('n sample to keep',n_sample_keep)

    return indices_to_keep, w_validation


def get_indices_each_node_opt(cfg,
                              indices_each_node_case,
                              indices_each_node_case_copy_init,
                              indices_to_keep=None):

    option = cfg.option

    if option == 1:
        n_sample_keep = len(keep_track_indices_dataset_train[0])
        indices_to_keep = np.arange(0, n_sample_keep)
        list_nodes_not_enough_data = []
        for n in range(0, cfg.n_nodes):
            indices_each_node_case[cfg.case][n] = select_intersection(
                indices_to_keep, indices_each_node_case_copy_init[cfg.case][n])
            # check if the number of point in a node is smaller than batch size
            if len(indices_each_node_case[cfg.case][n]) == 0:
                list_nodes_not_enough_data.append(n)

    elif option == 2:

        list_nodes_not_enough_data = []
        for n in range(0, cfg.n_nodes):
            indices_each_node_case[
                cfg.case][n] = indices_each_node_case_copy_init[cfg.case][n]
            # check if the number of point in a node is smaller than batch size
            if len(indices_each_node_case[cfg.case][n]) == 0:
                list_nodes_not_enough_data.append(n)

    elif option == 0:

        sampler_list = []
        train_indices_list = []
        list_nodes_not_enough_data = []
        for n in range(0, cfg.n_nodes):

            indices_each_node_case[cfg.case][n] = select_intersection(
                indices_to_keep, indices_each_node_case_copy_init[cfg.case][n])
            # check wehter the number of point in a node is smaller than batch size
            if len(indices_each_node_case[cfg.case][n]) == 0:
                list_nodes_not_enough_data.append(n)

    return indices_each_node_case, list_nodes_not_enough_data


def get_benchmark_dataset(sim, size_validation_set, cfg):

    #init
    label_missing_in_validation = [1000]
    while (len(label_missing_in_validation) != 0):

        train_indices_focus = keep_track_indices_dataset_train[0]
        train_indices_focus_copy = copy.deepcopy(train_indices_focus)
        np.random.seed(sim)
        np.random.shuffle(train_indices_focus_copy)
        list_indices_validation = train_indices_focus_copy[
            0:size_validation_set]
        validation_image = [
            train_image_all[i] for i in list_indices_validation
        ]
        validation_label = [
            train_label_all[i] for i in list_indices_validation
        ]

        sim += 1

        # split training into train and test
        #make sure validation contains each category of label
        lists_indices_per_label_validation = get_lists_indices_per_label(
            cfg.n_category, validation_label)
        label_missing_in_validation = get_label_missing(
            cfg.n_category, lists_indices_per_label_validation)

    return lists_indices_per_label_validation, validation_image, validation_label
