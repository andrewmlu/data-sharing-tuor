# Scenario , define target/ noise
dataset_mnist = 'MNIST_ORIG_ALL_LABELS'
dataset_fashion = 'MNIST_FASHION'
dataset_svnh_gray= 'SVNH_GRAY'
dataset_cifar10_gray = 'CIFAR_10_GRAY'
# always put dataset "focus" in first position, and the "noise" dataset after
dataset_list = [dataset_fashion,dataset_mnist,dataset_fashion,dataset_cifar10_gray,dataset_svnh_gray]



# Training in General
n_category = 10
model_name = 'ModelCNNMnist'
percentage_batch = 0.16
step_size = 0.01

# Federated Learning
total_iterations = 200000 #(equivalent to 20 000 iterations as set in the paper)
n_nodes = 10 # need to be larger or equal to 10


# Folder
dataset_file_path = 'datasets/dataset_files'
results_file_path = 'results'

