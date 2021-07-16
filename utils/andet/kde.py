import numpy as np
from joblib import Parallel, delayed
import sklearn.neighbors as neighbors


class KernelDensity(object):
    ''' Kernel Density for anomaly detection
    '''

    def __init__(self, kernel: str = 'gaussian', bandwidth: float = 0.2) -> None:
        super().__init__()
        self.bandwidth = bandwidth
        self.kernel = kernel

    @staticmethod
    def __calc_batch__(input_numpy: np.ndarray, bandwidth: float, kernel: str) -> np.ndarray:
        kde = neighbors.KernelDensity(bandwidth=bandwidth, kernel=kernel)
        kde = kde.fit(input_numpy)
        rtn = kde.score_samples(input_numpy)
        rtn = np.exp(rtn)
        return np.reshape(rtn, newshape=(1, -1))

    def __call__(self, input: np.ndarray) -> np.ndarray:
        ''' from memory key to memory value
        Args:
            input (np.ndarray): the input data points, [batch_size, memory_size]
        '''
        batch_size = input.shape[0]
        input_numpy = np.reshape(input, newshape=(batch_size, -1, 1))
       #rtns = Parallel(n_jobs=1)(
       #    delayed(KernelDensity.__calc_batch__)(input_numpy[idx, :, :], self.bandwidth, self.kernel)
       #    for idx in range(batch_size)
       #)
        rtns = [(input_numpy[idx, :, :], self.bandwidth, self.kernel) for idx in range(batch_size)]
        outputs = np.concatenate(rtns, axis=0)
        return outputs
