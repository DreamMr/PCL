import importlib
import torch.utils.data

class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, dataset,batch_size,shuffle=True,pin_memory=False,num_workers=2):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.dataset = dataset
        self.batch_size=batch_size
        self.dataset_size = len(dataset)
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            pin_memory = pin_memory,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.batch_size > self.dataset_size:
                break
            yield data