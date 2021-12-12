"""
This file should be migrated to a jupyter notebook.
"""

from classifier.network.alex_net import *
from classifier.plugin import *
from classifier.metric import *
from typing import Callable, Dict

import numpy as np
import torchvision
from torchvision import transforms, models

from classifier.metric import *
from classifier.network.vgg16_acon import *
from classifier.network.vgg16_metaacon import *
from classifier.network.vgg16_relu import *
from classifier.plugin import *

TRAINED_MODELS_PATH = Path("../drive/MyDrive/Colab Notebooks/COMP551/Project 4/vgg_exp_results")


def get_mean_std(cifar):
    features = [item[0] for item in cifar]
    features = torch.stack(features, dim=0)
    mean = features[..., 0].mean(), features[..., 1].mean(), features[..., 2].mean()
    std = features[..., 0].std(unbiased=False), features[..., 1].std(unbiased=False), features[..., 2].std(
        unbiased=False)
    return (mean, std)


def save_data(path: str = './dataset', val_proportion: float = 0.1, random_seed: int = 0):
    torch.manual_seed(random_seed)

    transform_train = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_original = torchvision.datasets.CIFAR100(root=path, train=True, transform=transform_train,
                                                   download=True)

    val_size = int(len(train_original) * val_proportion)
    train_size = len(train_original) - val_size

    train, _ = random_split(train_original, [train_size, val_size])

    mean_std = get_mean_std(train)
    print(mean_std)

    #mean_std = ((0.4992, 0.4992, 0.4992), (0.2891, 0.2891, 0.2891)) # 224 * 224
    #mean_std = ((0.4994, 0.4994, 0.4987), (0.2890, 0.2890, 0.2867)) # 128 * 128
    #mean_std = ((0.4996, 0.4983, 0.4947), (0.2888, 0.2847, 0.2818)) # 64 * 64
    # mean_std = ((0.4882, 0.4877, 0.4869), (0.2802, 0.2798, 0.2785)) # original 28 * 28

    transform_train = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*mean_std),
    ])

    train_normalized = torchvision.datasets.CIFAR100(root=path, train=True, transform=transform_train)

    train, val = random_split(train_normalized, [train_size, val_size])

    transform_test = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(*mean_std),
    ])

    test = torchvision.datasets.CIFAR100(root=path, train=False, transform=transform_test)

    torch.save(train, Path(Path(path)) / 'train')
    torch.save(val, Path(Path(path)) / 'val')
    torch.save(test, Path(Path(path)) / 'test')

def load_data(path: str = './dataset'):
    train = torch.load(Path(Path(path)) / 'train')
    val = torch.load(Path(Path(path)) / 'val')
    test = torch.load(Path(Path(path)) / 'test')
    return train, val, test


ADAM_PROFILE = OptimizerProfile(Adam, {
    "lr": 0.0005,
    "betas": (0.9, 0.99),
    "eps": 1e-8
})

SGD_PROFILE = OptimizerProfile(SGD, {
    'lr': 0.0005,
    'momentum': 0.99
})

# save_data()
TRAIN, VAL, TEST = load_data()


def train_model(model: Callable[..., Module], fname: str, model_params: Dict[str, Any] = {},
                epochs: int = 100,
                continue_from: int = 0,
                batch_size: int = 100):
    print(fname)
    print(model)
    print(model_params)

    model_path = Path(TRAINED_MODELS_PATH / fname)

    clf = NNClassifier(model, TRAIN, VAL, network_params=model_params)

    conv_params = sum(p.numel() for p in clf.network.features.parameters() if p.requires_grad)
    print(conv_params)

    print(f"Epochs to train: {epochs}")
    print(f"Continue from epoch: {continue_from}")
    if continue_from > 0:
        clf.load_network(model_path, continue_from)

    clf.set_optimizer(ADAM_PROFILE)

    clf.train(epochs,
               batch_size=batch_size,
               plugins=[
                   save_good_models(model_path),
                   calc_train_val_performance(accuracy),
                   print_train_val_performance(accuracy),
                   log_train_val_performance(accuracy),
                   save_training_message(model_path),
                   plot_train_val_performance(model_path, 'Modified AlexNet', accuracy, show=False,
                                              save=True),
                   elapsed_time(),
                   save_train_val_performance(model_path, accuracy),
               ],
               start_epoch=continue_from + 1
               )


def get_best_epoch(fname: str):
    """
    get the number of best epoch
    chosen from: simplest model within 0.001 acc of the best model
    :param fname:
    :return:
    """
    model_path = Path(TRAINED_MODELS_PATH / fname)
    performances = load_train_val_performance(model_path)
    epochs = performances['epochs']
    val = performances['val']
    highest = max(val)
    index_to_chose = -1
    for i in range(len(val)):
        if abs(val[i] - highest) < 0.001:
            index_to_chose = i
            print(f"Val acc of model chosen: {val[i]}")
            break
    return epochs[index_to_chose]

def obtain_test_acc(model: Callable[..., Module], fname: str, model_params: Dict[str, Any] = {}, *args, **kwargs):
    best_epoch = get_best_epoch(fname)
    clf = NNClassifier(model, None, None, network_params=model_params)
    model_path = Path(TRAINED_MODELS_PATH / fname)
    clf.load_network(model_path, best_epoch)
    acc = clf.evaluate(TEST, accuracy)
    # one-line from https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    print(f"\nTEST SET RESULT FOR {fname}: {acc}\n")


def train_and_test(model: Callable[..., Module], fname: str, model_params: Dict[str, Any] = {},
                   epochs: int = 100,
                   continue_from: int = 0,
                   batch_size: int = 100
                   ):
    train_model(model, fname, model_params, epochs, continue_from, batch_size)
    obtain_test_acc(model, fname, model_params)

def plot_valacc(entries: Dict[str, str], title: str, target: str, epochs_to_show: int = 50, show: bool = False):
    """

    :param entries: dict of the form {file_name: label}
    :param title: title of the plot
    :param target: target file name to save the plot
    :param epochs_to_show:
    :param show:
    :return:
    """
    plt.figure()
    for k in entries:
        model_path = Path(TRAINED_MODELS_PATH / k)
        performances = load_train_val_performance(model_path)
        index = -1
        epochs = performances['epochs']
        for i in epochs:
            if epochs[i] == epochs_to_show:
                index = i
                break
        epochs = epochs[:index]
        val = performances['val'][:index]
        plt.plot(epochs, val,
                 label=entries[k], alpha=0.5)
    # plt.ylim(bottom=0.5)
    plt.xlabel('Number of epochs')
    plt.ylabel('Validation accuracy')
    plt.title(title)
    plt.legend()
    plt.savefig(TRAINED_MODELS_PATH / target)
    if show:
        plt.show()


if __name__ == '__main__':
    params = {'num_classes': 100}
    original = (64, 192, 384, 256, 256, 4096)
    s1 = (48, 144, 288, 192, 192, 3072)
    s2 = (32, 96, 192, 128, 128, 2048)
    to_run = (
        # (models.efficientnet_b0, 'effnet-relu', params),
        # (models.shufflenet_v2_x0_5, 'shuffle-relu', params),

        # (AlexNet, 'alex-metaacon', {'activation': 'metaacon'}),
        # (AlexNet, 'alex-acon', {'activation': 'acon'}),
        # (AlexNet, 'alex-relu', {'activation': 'relu'}),
        # (AlexNet, 'alex-metaacon-s1', {'activation': 'metaacon', 'sizes': s1}, 50),
        # (AlexNet, 'alex-acon-s1', {'activation': 'acon', 'sizes': s1}, 50),
        # (AlexNet, 'alex-relu-s1', {'activation': 'relu', 'sizes': s1}, 50),
        (AlexNet, 'alex-metaacon-s2', {'activation': 'metaacon', 'sizes': s2}, 50),
        (AlexNet, 'alex-acon-s2', {'activation': 'acon', 'sizes': s2}, 50),
        (AlexNet, 'alex-relu-s2', {'activation': 'relu', 'sizes': s2}, 50),
    )
    for p in to_run:
        train_and_test(*p)

    entries = {
        'alex-metaacon-s2': 'MetaACON',
        'alex-acon-s2': 'ACON',
        'alex-relu-s2': 'ReLU'
    }
    plot_valacc(entries, 'AlexNet s2', 'alex-s2.jpg', 100)