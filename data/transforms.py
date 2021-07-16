import torch
import numpy as np
import torch.nn as nn
from typing import Tuple


class Compose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):

    def __init__(self):
        pass

    def __call__(self, input: np.ndarray) -> torch.Tensor:
        ''' np.ndarray to torch.tensor
        '''
        #import pdb;pdb.set_trace()
        if input.dtype == np.float64:
            input = input.astype(np.float32)
        if input.dtype == np.int32:
            input = input.astype(np.int64)

        # preloding dataset in GPU
        if torch.is_tensor(input):
            return input
        return torch.from_numpy(input)

class Normalize(nn.Module):

    def __init__(self, mean: Tuple = None, std: Tuple = None) -> None:
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        ''' normalize the ndarray
        '''
        dtype = input.dtype

        mean = torch.as_tensor(self.mean, dtype=dtype, device=input.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=input.device)

        input.sub_(mean).div_(std)
        return input


def paddingZeros(input: torch.Tensor, padded_size: int, dtype: torch.AnyType, device: torch.device, tensor: bool = True) -> torch.Tensor:
    ''' Padding zeros for getting padded_size
    Args:
        input (torch.Tensor): the tensor before padded
        padded_size (int): the size of padded tensor
    '''
    if input is None and tensor:
        return torch.zeros(
            size=(padded_size,),
            dtype=dtype
        ).to(device)

    if input is None and not tensor:
        return np.zeros(shape=(padded_size,), dtype=dtype)

    input_size = input.shape[0]
    if input_size == padded_size:
        return input

    if torch.is_tensor(input):
        padded_tensor = torch.zeros(
            size=(padded_size - input_size,),
            dtype=dtype
        ).to(input.device)
        return torch.cat([padded_tensor, input], dim=0)

    padded_array = np.zeros(
        shape=(padded_size - input_size,),
        dtype=dtype
    )

    return np.concatenate((padded_array, input), axis=0)
