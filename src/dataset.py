from torchvision import datasets
from torchvision.transforms import ToTensor


def get_mnist(dir:str = 'data') :
    training_data = datasets.MNIST(
        root="data",
        train= True,
        download=True,
        transform=ToTensor()
        
    )
    
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    
    return training_data, test_data

