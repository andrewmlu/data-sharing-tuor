# Scenario , define target/ noise
dataset_mnist = 'MNIST_ORIG_ALL_LABELS_RGB'
dataset_fashion = 'MNIST_FASHION_RGB'
dataset_svnh= 'SVNH'
dataset_cifar10 = 'CIFAR_10'
# always put dataset "focus" in first position, and the "noise" dataset after
dataset_list = [dataset_svnh,dataset_cifar10,dataset_mnist,dataset_fashion]



# Training in General
n_category = 10
model_name = 'ModelCNNCifar10'
percentage_batch = 0.16
step_size = 0.01

# Federated Learning
total_iterations = 200000 #(equivalent to 20 000 iterations as set in the paper)
n_nodes = 10 # need to be larger or equal to 10


# Folder
dataset_file_path = 'datasets/dataset_files'
results_file_path = 'results'

