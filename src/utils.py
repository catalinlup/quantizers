from typing import Union
import h5py
import numpy as np

def sample_from_dataset_vectors(dataset: h5py.Dataset, sample_size: Union[int, None]) -> np.ndarray:
    """
    Takes a random sample from the provided h5 dataset. If the sample size is none, raturns the entire dataset.
    """

    if sample_size == None:
        return dataset[:, :]
    

    random_ids = np.random.choice(dataset.shape[0], size=sample_size, replace=False)
    random_ids.sort()
    random_sample = dataset[random_ids]

    return random_sample