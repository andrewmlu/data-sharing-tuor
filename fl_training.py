import copy
import time
import numpy as np
from util.sampling import MinibatchSampling
from util.time_generation import TimeGeneration


def federated_training(cfg, model, train_image_all, train_label_all, sim,
                       list_nodes_not_enough_data, indices_each_node_case,
                       train_image_focus, train_label_focus, train_image_noise,
                       train_label_noise, test_image_focus, test_label_focus,
                       test_image_noise, test_label_noise, validation_image,
                       validation_label, w_validation):
    with open(cfg.results_file_path + '/loss-accu.csv', 'a') as f:
        f.write(
            'iterations, approach, loss_train_focus, loss_train_noise, accuracy_test_focus, accuracy_test_noise, accuracy_test_validation'
            + '\n')
        f.close()
    cfg.time_gen = TimeGeneration(1.0, 0.0, 1e-10, 0.0, 0.0, 0.0)
    cfg.local_iterations_between_global_averaging_all = [10]
    cfg.max_time = int(cfg.total_iterations/cfg.local_iterations_between_global_averaging_all[0])#
    cfg.plot_steps = 1



    for tau_setup in cfg.local_iterations_between_global_averaging_all:

        dim_w = model.get_weight_dimension(train_image_all, train_label_all)
        w_global_init = model.get_init_weight(dim_w, rand_seed=sim)
        w_global = copy.deepcopy(w_global_init)
        total_time = 0

        sampler_list = []
        train_indices_list = []
        w_list = []

        save_batch = []
        # initialization of each client
        for n in range(0, cfg.n_nodes):

            if n not in list_nodes_not_enough_data:
                indices_this_node = indices_each_node_case[cfg.case][n]
                batch_size = int(
                    round(len(indices_this_node) * cfg.percentage_batch))
                save_batch.append(batch_size)

                if batch_size == 0:
                    batch_size = 1
                sampler = MinibatchSampling(indices_this_node, batch_size, sim)
                train_indices = None  # To be defined later
                sampler_list.append(sampler)
                train_indices_list.append(train_indices)
                w_list.append(copy.deepcopy(w_global_init))
                # w_validation = w_global_init
            else:
                save_batch.append(None)
                train_indices = None  # To be defined later
                sampler_list.append(None)
                train_indices_list.append(train_indices)
                w_list.append(copy.deepcopy(w_global_init))
                # w_validation = w_global_init
        w_validation_traning = copy.deepcopy(w_global_init)

        steps = 0


        np.save(cfg.results_file_path + '/save_batch.npy', save_batch)

        sup_l = []
        while True:


            w_global_prev = w_global
            l = []
            for n in range(0, cfg.n_nodes):

                if n not in list_nodes_not_enough_data:

                    starting = time.time()

                    model.start_consecutive_training(w_init=w_list[n])
                    for i in range(0, tau_setup):
                        # train_indices = train_indices_list[n]
                        sample_indices = sampler_list[n].get_next_batch()
                        train_indices = sample_indices

                        model.run_one_step_consecutive_training(
                            train_image_all, train_label_all, train_indices)

                    w_list[n] = model.end_consecutive_training_and_get_weights(
                    )
                    l.append(time.time() - starting)
                    # print('Average',sum(l)/len(l))

            sup_l.append(sum(l) / len(l))
            #print('batch size', batch_size)
            #print('Sup average', sum(sup_l) / len(sup_l))

            model.start_consecutive_training(w_init=w_validation_traning)
            model.run_one_step_consecutive_training(
                validation_image, validation_label,
                np.arange(0, len(validation_label)))
            w_validation_traning = model.end_consecutive_training_and_get_weights(
            )

            w_global = np.zeros(dim_w)
            # print('reinitialisation')
            total_data_size = 0

            for n in range(0, cfg.n_nodes):
                if n not in list_nodes_not_enough_data:
                    total_data_size += len(indices_each_node_case[cfg.case][n])
                    w_global += len(
                        indices_each_node_case[cfg.case][n]) * w_list[n]

            # add the validation
            total_data_size += len(validation_label)
            w_global += len(validation_label) * w_validation_traning

            w_global /= total_data_size

            if True in np.isnan(w_global):
                print('*** w_global is NaN, using previous value')
                w_global = w_global_prev  # If current w_global contains NaN value, use previous w_global

            steps += 1

            if steps >= cfg.plot_steps:
                steps = 0
            #if total_time % 50 == 0:

                loss_value_train_focus = model.loss(train_image_focus,
                                                    train_label_focus,
                                                    w_global)
                loss_value_train_noise = model.loss(train_image_noise,
                                                    train_label_noise,
                                                    w_global)
                accu_value_test_focus = model.accuracy(test_image_focus,
                                                       test_label_focus,
                                                       w_global)
                accu_value_test_noise = model.accuracy(test_image_noise,
                                                       test_label_noise,
                                                       w_global)

                if cfg.option == 0:
                    accu_value_val = model.accuracy(test_image_focus,
                                                    test_label_focus,
                                                    w_validation)

                if cfg.option == 0:
                    with open(cfg.results_file_path + '/loss-accu.csv',
                              'a') as f:
                        f.write(
                            str(total_time) + ',' + str(cfg.option) + ',' +
                            str(loss_value_train_focus) + ',' +
                            str(loss_value_train_noise) + ',' +
                            str(accu_value_test_focus) + ',' +
                            str(accu_value_test_noise) + ','+ str(accu_value_val) +
                            '\n')
                        f.close()

                else:
                    with open(cfg.results_file_path + '/loss-accu.csv',
                              'a') as f:
                        f.write(
                            str(total_time) + ',' + str(cfg.option) + ',' +
                            str(loss_value_train_focus) + ',' +
                            str(loss_value_train_noise) + ',' +
                            str(accu_value_test_focus) + ',' +
                            str(accu_value_test_noise) + '\n')
                        f.close()

            for n in range(0, cfg.n_nodes):
                w_list[n] = copy.deepcopy(w_global)
            w_validation_traning = copy.deepcopy(w_global)

            if isinstance(cfg.time_gen, (list, )):
                t_g = cfg.time_gen[cfg.case]
            else:
                t_g = cfg.time_gen

            it_each_local = max(0.00000001,
                                np.sum(t_g.get_local(tau_setup)) / tau_setup)
            it_each_global = t_g.get_global(1)[0]

            # Compute number of iterations is current slot
            total_time += it_each_local * tau_setup + it_each_global


            if total_time >= cfg.max_time:
                break
