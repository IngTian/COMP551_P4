"""
This file should be migrated to a jupyter notebook.
"""
from classifier.network.alex_net_leakyrelu import *
from classifier.network.alex_net_acon import *
from classifier.network.alex_net_metaacon import *
from classifier.network.alex_net_metaacon1 import *
from classifier.network.alex_net_relu import *
from classifier.network.vgg16_acon import *
from classifier.network.vgg16_relu import *
from classifier.network.vgg16_metaacon import *
from classifier.plugin import *
from classifier.metric import *
import torchvision
from torchvision import transforms

TRAINED_MODELS_PATH = Path("../drive/MyDrive/Colab Notebooks/COMP551/Project 4/vgg_exp_results")


def get_mean_std(cifar):
    features = [item[0] for item in cifar]
    features = torch.stack(features, dim=0)
    mean = features[...,0].mean(), features[...,1].mean(), features[...,2].mean()
    std = features[...,0].std(unbiased=False), features[...,1].std(unbiased=False), features[...,2].std(unbiased=False)
    return (mean, std)


def save_data(path: str = './dataset', val_proportion: float = 0.1, random_seed: int = 0):
    torch.manual_seed(random_seed)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.ToTensor(),
    ])

    train_original = torchvision.datasets.CIFAR100(root='./dataset/', train=True, transform=transform_train,
                                                   download=True)

    val_size = int(len(train_original) * val_proportion)
    train_size = len(train_original) - val_size

    train, _ = random_split(train_original, [train_size, val_size])

    # mean_std = get_mean_std(train)
    # print(mean_std)
    mean_std = ((0.4882, 0.4877, 0.4869), (0.2802, 0.2798, 0.2785))

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(*mean_std)
    ])

    train_normalized = torchvision.datasets.CIFAR100(root='./dataset/', train=True, transform=transform_train)

    train, val = random_split(train_normalized, [train_size, val_size])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*mean_std)
    ])

    test = torchvision.datasets.CIFAR100(root='./dataset/', train=False, transform=transform_test)

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

#save_data()
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

    if continue_from > 0:
        clf.load_network(model_path, continue_from)

    clf.set_optimizer(ADAM_PROFILE)

    clf.train(epochs,
               batch_size=batch_size,
               plugins=[
                   save_model(model_path, save_last=False),
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
    chosen from: simplest model within 1 std of acc of the best model
    :param fname:
    :return:
    """
    model_path = Path(TRAINED_MODELS_PATH / fname)
    performances = load_train_val_performance(model_path)
    epochs = performances['epochs']
    val = performances['val']
    std = torch.std(Tensor(val), unbiased=False)
    highest = max(val)
    index_to_chose = -1
    for i in range(len(val)):
        if abs(val[i] - highest) < std:
            index_to_chose = i
            break
    return epochs[index_to_chose]

def obtain_test_acc(model: Callable[..., Module], fname: str, model_params: Dict[str, Any] = {}):
    best_epoch = get_best_epoch(fname)
    clf = NNClassifier(model, None, None, network_params=model_params)
    model_path = Path(TRAINED_MODELS_PATH / fname)
    clf.load_network(model_path, best_epoch)
    acc = clf.evaluate(TEST, accuracy)
    print(f"\nTEST SET RESULT: {acc}")

def train_and_test(model: Callable[..., Module], fname: str, model_params: Dict[str, Any] = {},
                   epochs: int = 100,
                   continue_from: int = 0,
                   batch_size: int = 100
                   ):
    train_model(model, fname, model_params, epochs, continue_from, batch_size)
    obtain_test_acc(model, fname, model_params)

if __name__ == '__main__':
    alex_params = {'n_way': 2, 'depth': (1, 1, 2)}

    train_and_test(VGG16_ACON, 'vgg16-acon')

    train_and_test(VGG16_Meta_ACON, 'vgg16-meta-acon')

    train_and_test(VGG16, 'vgg16-vanilla')
