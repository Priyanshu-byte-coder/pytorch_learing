
#script mode
#usign jupyter magic to save the cell as a .py script

from torchvision import datasets,transforms
from torch.utils.data import DataLoader

NUM_WORKERS=4
def create_dataloaders(
    train_dir:str,
    test_dir:str,
    transform:transforms.Compose,
    batch_size:int,
    num_workers:int=NUM_WORKERS):
    """takes in training directory and testing directory path and turns them into pytorch datasets and then into pytorch dataloaders
    Args:
        train_dir:path to training directory,
        test_dir:path to testing directory,
        transform:torchvision transforms to perform on training and testing data,
        batch_size:Number of samples per batrch in each of the dataloader,
        num_workers:An integer for number of workers per dataloader

    Returns:
        A tuple of (train_dataloader,test_dataloader,class_names)
    """
    train_data=datasets.ImageFolder(train_dir,transform=transform)
    test_data=datasets.ImageFolder(test_dir,transform=transform)

    class_names=train_data.classes

    train_dataloader=DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_dataloader=DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_dataloader,test_dataloader,class_names

