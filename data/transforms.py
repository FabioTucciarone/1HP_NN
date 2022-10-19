"""
Definition of problem-specific transform classes
"""

import logging
import numpy as np
from torchvision.transforms import Normalize
import torch
from data.utils import PhysicalVariables

def compute_data_max_and_min(data):
    """
    Calculate the per-channel data maximum and minimum value of a given data point
    :param data: numpy array of shape (Nx)CxHxWxD
        (for N data points with C channels of spatial size HxWxD)
    :returns: per-channels mean and std; numpy array of shape C
    """
    # does not check the input type
    for n in data:
        assert type(n) == np.ndarray, "Data is not a numpy array"
    # TODO does this produce/assert the correct output?

    max, min = None, None

    # Calculate the per-channel max and min of the data
    max = np.max(data, axis=(1,2,3), keepdims=True)
    min = np.min(data, axis=(1,2,3), keepdims=True)

    return max, min
    
class RescaleTransform:
    """Transform class to rescale data to a given range"""
    def __init__(self, out_range=(0, 1)):
        """
        :param out_range: Value range to which data should be rescaled to
        :param in_range:  Old value range of the data
            e.g. (0, 255) for images with raw pixel values
        """
        self.min = out_range[0]
        self.max = out_range[1]

    def __call__(self, data, **kwargs):
        print("rescale")
        # calc max, min of data for each channel of one datapoint, broadcast
        self._data_max, self._data_min = compute_data_max_and_min(data)
        # self._data_max = self._data_max[:,np.newaxis,np.newaxis,np.newaxis]
        # self._data_min = self._data_min[:,np.newaxis,np.newaxis,np.newaxis]

        # rescale data to new range
        data = data - self._data_min  # normalize to (0, data_max-data_min)
        data /= (self._data_max - self._data_min)  # normalize to (0, 1)
        data *= (self.max - self.min)  # norm to (0, target_max-target_min)
        data += self.min  # normalize to (target_min, target_max)

        return data

class NormalizeTransform:
    """
    Transform class to normalize data using mean and std
    Functionality depends on the mean and std provided in __init__():
        - if mean and std are single values, normalize the entire image
        - if mean and std are numpy arrays of size C for C image channels,
            then normalize each image channel separately
    """

    def __init__(self, reduced_to_2D:bool=False):
        """
        :param mean: mean of data to be normalized
            can be a single value, or a numpy array of size C
        :param std: standard deviation of data to be normalized
             can be a single value or a numpy array of size C
        """
        # self.mean = None
        # self.std = None
        self.reduced_to_2D = reduced_to_2D

    def __call__(self, data:PhysicalVariables): #, index_material_id=None):
        # manual shift of material IDs  TODO change this in pflotran file!! not here
        name_material = "Material_ID"
        if name_material in data.keys():
            data[name_material].value -= 1
            mask = data[name_material].value == 2
            data[name_material].value[mask] = -1
        ## LEGACY
        # if self.reduced_to_2D:
        #     if index_material_id:
        #         data[index_material_id,:,:] -= 1
        #         # only required, if extraction well (with ID=3) exists
        #         mask = data[index_material_id,:,:] == 2
        #         data[index_material_id,:,:][mask] = -1
        # else:
        #     if index_material_id:
        #         data[index_material_id,:,:,:] -= 1
        #         # only required, if extraction well (with ID=3) exists
        #         mask = data[index_material_id,:,:,:] == 2
        #         data[index_material_id,:,:,:][mask] = -1

        # bool_just_one_input=False
        # # ugly workaround for just one input variable
        # if data.shape[0] == 1: # if only one input variable like Material_ID
        #     bool_just_one_input = True
        #     temp_zeros = torch.zeros(data.shape)
        #     data = torch.cat([data, temp_zeros], dim=0)

        ## TODO REQUIRED??
        # if len(data) == 1:
        #     temp_zeros = torch.zeros(data.shape)
        #     data = torch.cat([data, temp_zeros], dim=0)

        # if self.reduced_to_2D:
        #     # calc mean, std per channel then normalize data to mean and std, including broadcasting
        #     self.mean = data.mean(dim=(1, 2), keepdim=True)
        #     self.std = data.std(dim=(1, 2), keepdim=True)
        # else:
        #     # calc mean, std per channel then normalize data to mean and std, including broadcasting
        #     self.mean = data.mean(dim=(1, 2, 3), keepdim=True)
        #     self.std = data.std(dim=(1, 2, 3), keepdim=True)

        # # ugly workaround for just one input variable
        # if bool_just_one_input:
        #     self.mean = torch.tensor_split(self.mean, 2)[0]
        #     self.std = torch.tensor_split(self.std, 2)[0]
        #     data = torch.tensor_split(data, 2)[0]
        
        for prop in data.keys():
            # calc mean, std per channel then normalize data to mean and std, including broadcasting
            data[prop].calc_mean() # dim=(1, 2, 3), keepdim=True)
            data[prop].calc_std() # dim=(1, 2, 3), keepdim=True)

        # normalize data to mean and std
        for prop in data.keys():
            Normalize(data[prop].mean, data[prop].std, inplace=True)(data[prop].value)
            # squeeze in case of reduced_to_2D_wrong necessary because unsqueezed before for Normalize to work
            data[prop].value = torch.squeeze(data[prop].value)
        # data = Normalize(self.mean, self.std, inplace=True)(data)

        # assert if rounded value of mean is not 0
        # assert torch.abs(torch.round(data.mean(dim=(0,1,2)),decimals=12)) == 0
        # assert torch.abs(torch.round(data.std(dim=(0,1,2)),decimals=12)) <= 1
        # assert torch.abs(torch.round(data.std(dim=(0,1,2)),decimals=12)) >= 0.999
        return data

    def reverse_OLD_FORMAT(self, data, mean=None, std=None, index_material_id=None):
        # reverse normalization
        # data = torch.from_numpy(data) * self.std + self.mean # version if std, mean in class
        data = torch.add(torch.mul(data,std), mean)
        #TODO why is data numpy format??
        
        # manual shift of material IDs  TODO change this in pflotran file!! not here
        if self.reduced_to_2D:
            if index_material_id:
                mask = data[index_material_id,:,:] == -1

                # only required, if extraction well (with ID=3) exists
                data[index_material_id,:,:][mask] = 2
                data[index_material_id,:,:] += 1
        else:
            if index_material_id:
                mask = data[index_material_id,:,:,:] == -1

                # only required, if extraction well (with ID=3) exists
                data[index_material_id,:,:,:][mask] = 2
                data[index_material_id,:,:,:] += 1

        return data #, data.mean(dim=(1, 2, 3), keepdim=True), data.std(dim=(1, 2, 3), keepdim=True)

class PowerOfTwoTransform: #CutOffEdgesTransform:
    """
    Transform class to reduce dimensionality to be a power of 2
    (TODO ?? data cleaning: cut of edges - to get rid of problems with boundary conditions)
        # cut off edges of images with unpretty boundary conditions
        # problem?: "exponential" behaviour at boundaries??
    """
    def __init__(self, oriented="center"):
        self.orientation = oriented

    def __call__(self, data, **kwargs):
        
        def po2(array, axis):
            dim = array.shape[axis]
            target = 2 ** int(np.log2(dim))
            delta = (dim - target)//2
            if self.orientation == "center":
                result = np.take(array, range(delta, target+delta), axis=axis)
            elif self.orientation == "left": # cut off only from right
                result = np.take(array, range(0, target), axis=axis)
            return result
        
        for prop in data.keys():
            for axis in (0,1,2): #,3): # for axis width, length, depth
                data[prop].value = po2(data[prop].value, axis)

        return data
        #return data_np[:,1:-1,1:-3,1:-1]

class ReduceTo2DTransform:
    """
    Transform class to reduce data to 2D, reduce in x, in height of hp: x=7 (if after Normalize)
    #TODO still np?
    """
    def __init__(self, reduce_to_2D_wrong=False):
        # if reduce_to_2D_wrong then the data will still be reduced to 2D but in x,y dimension instead of y,z
        self.reduce_to_2D_wrong = reduce_to_2D_wrong

    def __call__(self, data, x=9, **kwargs):
        # reduce data to 2D
        if self.reduce_to_2D_wrong:
            x = 9
            for prop in data.keys():
                data[prop].value.transpose_(1,3)

        for prop in data.keys():
            assert x <= data[prop].value.shape[0], "x is larger than data dimension 0"
            data[prop].value = data[prop].value[x,:,:]
            data[prop].value = torch.unsqueeze(data[prop].value,0)
        logging.info("Reduced data to 2D, but still has dummy dimension 0 for Normalization to work")
        return data #[:,x,:,:]

class ToTensorTransform:
    """Transform class to convert np.array-data to torch.Tensor"""
    def __init__(self):
        pass
    def __call__(self, data, **kwargs):
        for prop in data.keys():
            data[prop].value = torch.from_numpy(data[prop].value)
        return data

class ComposeTransform:
    """Transform class that combines multiple other transforms into one"""
    def __init__(self, transforms):
        """
        :param transforms: transforms to be combined
        """
        self.transforms = transforms
        # self.mean = None
        # self.std = None

    def __call__(self, data, **kwargs):
        for transform in self.transforms:
            data = transform(data, **kwargs)
        #     try:
        #         self.mean = transform.mean
        #         self.std = transform.std
        #     except:
        #         print(f"Transform {transform} didn' work")
        return data #, self.mean, self.std

    def reverse_OLD_FORMAT(self, data, **kwargs):
        for transform in reversed(self.transforms):
            try:
                data = transform.reverse(data, **kwargs)
            except AttributeError:
                pass
                #print(f"for transform {transform} no reverse implemented")
        return data
