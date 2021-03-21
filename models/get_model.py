import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_model(model_class_name):
    if model_class_name == 'ModelCNNMnist':
        from models.cnn_mnist import ModelCNNMnist
        return ModelCNNMnist()
    elif model_class_name == 'ModelCNNMnist_Binary':
        from models.cnn_mnist_binary import ModelCNNMnistBinary
        return ModelCNNMnistBinary()
    elif model_class_name == 'ModelCNNMnistSimple':
        from models.cnn_mnist_simple import ModelCNNMnistSimple
        return ModelCNNMnistSimple()
    elif model_class_name == 'ModelCNNMnistFc':
        from models.cnn_mnist_fc import ModelCNNMnistFc
        return ModelCNNMnistFc()
    elif model_class_name == 'ModelCNNMnistFc3Layer':
        from models.cnn_mnist_fc_3layer import ModelCNNMnistFc3Layer
        return ModelCNNMnistFc3Layer()
    elif model_class_name == 'ModelCNNEmnist':
        from models.cnn_emnist import ModelCNNEmnist
        return ModelCNNEmnist()
    elif model_class_name == 'ModelCNNEmnistSimple':
        from models.cnn_emnist_simple import ModelCNNEmnistSimple
        return ModelCNNEmnistSimple()
    elif model_class_name == 'ModelNNSuperposition':
        from models.nn_superposition import ModelNNSuperposition
        return ModelNNSuperposition()
    elif model_class_name == 'ModelNNSuperposition2':
        from models.nn_superposition2 import ModelNNSuperposition2
        return ModelNNSuperposition2()
    elif model_class_name == 'ModelCNNCifar10':
        from models.cnn_cifar10 import ModelCNNCifar10
        return ModelCNNCifar10()
    elif model_class_name == 'ModelCNNCifar10Simple':
        from models.cnn_cifar10_simple import ModelCNNCifar10Simple
        return ModelCNNCifar10Simple()
    elif model_class_name == 'ModelKmeans':
        from models.k_means import ModelKmeans
        return ModelKmeans()
    elif model_class_name == 'ModelLinearRegression':
        from models.linear_regression import ModelLinearRegression
        return ModelLinearRegression()
    elif model_class_name == 'ModelSVM':
        from models.svm import ModelSVM
        return ModelSVM()
    elif model_class_name == 'ModelSVMSmooth':
        from models.svm_smooth import ModelSVMSmooth
        return ModelSVMSmooth()

    else:
        raise Exception("Unknown model class name")
