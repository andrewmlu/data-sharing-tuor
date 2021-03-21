#import configs.config as cfg
from models.get_model import get_model
from util.preprocessing import multiple_to_single_dataset, get_indices_each_node_opt, get_benchmark_dataset, split_focus_noise, train_validation
from fl_training import federated_training
import copy
from argparse import ArgumentParser
import importlib
import os

parser = ArgumentParser()


parser.add_argument('--config', '-c', required=True)
parser.add_argument('--approach', '-appr',required=True)
parser.add_argument('--samples_per_dataset', '-n',required=True)
parser.add_argument('--pc_validation', '-pc',required=True)
parser.add_argument('--sim', '-s',required=True)


def main():

    args = parser.parse_args()
    cfg = importlib.import_module('configs.' + args.config)
    cfg.option = int(args.approach)
    cfg.total_data = int(args.samples_per_dataset)
    cfg.perc_validation_set = float(args.pc_validation)
    cfg.dataset_focus = cfg.dataset_list[0]




    if not os.path.exists(cfg.results_file_path):
        os.mkdir(cfg.results_file_path)

    model = get_model(cfg.model_name)

    if hasattr(model, 'create_graph'):
        model.create_graph(learning_rate=cfg.step_size)

    # Combine all datasets in one single dataset,

    train_image_all, train_label_all, keep_track_indices_dataset_train, keep_track_indices_dataset_test, indices_each_node_case = multiple_to_single_dataset(
        cfg)

    indices_each_node_case_copy_init = copy.deepcopy(indices_each_node_case)
    dim_w = model.get_weight_dimension(train_image_all, train_label_all)

    # Split dataset into the focus part and the noise part, ( this is useful when running different baselines )

    train_image_focus, train_label_focus, test_image_focus, test_label_focus, train_image_noise, train_label_noise, test_image_noise, test_label_noise = split_focus_noise(
        keep_track_indices_dataset_train, keep_track_indices_dataset_test)

    size_validation_set = int(cfg.perc_validation_set * len(train_label_focus))

    for sim in range(0,int(args.sim)):
        # Obtain benchmark (validation) dataset
        lists_indices_per_label_validation, validation_image, validation_label = get_benchmark_dataset(
            sim, size_validation_set, cfg)

        # Train benchmark (validation) dataset

        indices_to_keep, w_validation = train_validation(
            indices_each_node_case_copy_init, model, dim_w, sim, cfg,
            lists_indices_per_label_validation, validation_image, validation_label)

        # Filter the data with the selected data (according to the approach selected in the config (option: 0,1,2,3)
        indices_each_node_case, list_nodes_not_enough_data = get_indices_each_node_opt(
            cfg, indices_each_node_case, indices_each_node_case_copy_init,
            indices_to_keep)

        # Start the federated training

        federated_training(cfg, model, train_image_all, train_label_all, sim,
                           list_nodes_not_enough_data, indices_each_node_case,
                           train_image_focus, train_label_focus, train_image_noise,
                           train_label_noise, test_image_focus, test_label_focus,
                           test_image_noise, test_label_noise, validation_image,
                           validation_label, w_validation)

if __name__ == '__main__':
    main()