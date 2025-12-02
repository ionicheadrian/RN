from torchvision.datasets import MNIST

train_transforms = None
test_transforms = None 
train_dataset = MNIST('data', train=True, transform=train_transforms, download=True)
test_dataset = MNIST('data', train=False, transform=test_transforms, download=True)
