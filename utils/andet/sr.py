''' https://docs.seldon.io/projects/alibi-detect/en/stable/methods/sr.html
'''
import copy
import logging
import numpy as np
from typing import Dict
from abc import ABC, abstractmethod
from joblib import Parallel, delayed


logger = logging.getLogger(__name__)
EPSILON = 1e-8
DEFAULT_META = {
    "name": None,
    "detector_type": None,  # online or offline
    "data_type": None  # tabular, image or time-series
}  # type: Dict


def outlier_prediction_dict():
    data = {
        'instance_score': None,
        'feature_score': None,
        'is_outlier': None
    }
    return copy.deepcopy({"data": data, "meta": DEFAULT_META})


class BaseDetector(ABC):
    """ Base class for outlier detection algorithms. """

    def __init__(self):
        self.meta = copy.deepcopy(DEFAULT_META)
        self.meta['name'] = self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__

    @property
    def meta(self) -> Dict:
        return self._meta

    @meta.setter
    def meta(self, value: Dict):
        if not isinstance(value, dict):
            raise TypeError('meta must be a dictionary')
        self._meta = value

    @abstractmethod
    def score(self, X: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray):
        pass


class ThresholdMixin(ABC):
    @abstractmethod
    def infer_threshold(self, X: np.ndarray) -> None:
        pass


class SpectralResidual(BaseDetector, ThresholdMixin):

    def __init__(self,
                 threshold: float = None,
                 window_amp: int = None,
                 window_local: int = None,
                 n_est_points: int = None,
                 n_grad_points: int = 5,
                 ) -> None:
        """
        Outlier detector for time-series data using the spectral residual algorithm.
        Based on "Time-Series Anomaly Detection Service at Microsoft" (Ren et al., 2019)
        https://arxiv.org/abs/1906.03821

        Parameters
        ----------
        threshold
            Threshold used to classify outliers. Relative saliency map distance from the moving average.
        window_amp
            Window for the average log amplitude.
        window_local
            Window for the local average of the saliency map.
        n_est_points
            Number of estimated points padded to the end of the sequence.
        n_grad_points
            Number of points used for the gradient estimation of the additional points padded
            to the end of the sequence.
        """
        super().__init__()

        if threshold is None:
            logger.warning('No threshold level set. Need to infer threshold using `infer_threshold`.')

        self.threshold = threshold
        self.window_amp = window_amp
        self.window_local = window_local
        self.conv_amp = np.ones((1, window_amp)).reshape(-1,) / window_amp
        self.conv_local = np.ones((1, window_local)).reshape(-1,) / window_local
        self.n_est_points = n_est_points
        self.n_grad_points = n_grad_points

        # set metadata
        self.meta['detector_type'] = 'online'
        self.meta['data_type'] = 'time-series'

    def infer_threshold(self,
                        X: np.ndarray,
                        t: np.ndarray = None,
                        threshold_perc: float = 95.
                        ) -> None:
        """
        Update threshold by a value inferred from the percentage of instances considered to be
        outliers in a sample of the dataset.

        Parameters
        ----------
        X
            Batch of instances.
        threshold_perc
            Percentage of X considered to be normal based on the outlier score.
        """
        if t is None:
            t = np.arange(X.shape[0])

        # compute outlier scores
        iscore = self.score(X, t)

        # update threshold
        self.threshold = np.percentile(iscore, threshold_perc)

    def saliency_map(self, X: np.ndarray) -> np.ndarray:
        """
        Compute saliency map.

        Parameters
        ----------
        X
            Time series of instances.

        Returns
        -------
        Array with saliency map values.
        """
        fft = np.fft.fft(X)
        amp = np.abs(fft)
        log_amp = np.log(amp + EPSILON)
        phase = np.angle(fft)
        ma_log_amp = np.convolve(log_amp, self.conv_amp, 'same')
        res_amp = log_amp - ma_log_amp
        sr = np.abs(np.fft.ifft(np.exp(res_amp + 1j * phase)))
        return sr

    def compute_grads(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Slope of the straight line between different points of the time series
        multiplied by the average time step size.

        Parameters
        ----------
        X
            Time series of instances.
        t
            Time steps.

        Returns
        -------
        Array with slope values.
        """
        dX = X[-1] - X[-self.n_grad_points-1:-1]
        dt = t[-1] - t[-self.n_grad_points-1:-1]
        mean_grads = np.mean(dX / dt) * np.mean(dt)
        return mean_grads

    def add_est_points(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Pad the time series with additional points since the method works better if the anomaly point
        is towards the center of the sliding window.

        Parameters
        ----------
        X
            Time series of instances.
        t
            Time steps.

        Returns
        -------
        Padded version of X.
        """
        grads = self.compute_grads(X, t)
        X_add = X[-self.n_grad_points] + grads
        X_pad = np.concatenate([X, np.tile(X_add, self.n_est_points)])
        return X_pad

    def score(self, X: np.ndarray, t: np.ndarray = None) -> np.ndarray:
        """
        Compute outlier scores.

        Parameters
        ----------
        X
            Time series of instances.
        t
            Time steps.

        Returns
        -------
        Array with outlier scores for each instance in the batch.
        """
        if t is None:
            t = np.arange(X.shape[0])

        if len(X.shape) == 2:
            n_samples, n_dim = X.shape
            X = X.reshape(-1,)
            if X.shape[0] != n_samples:
                raise ValueError('Only univariate time series allowed for SR method. Number of features '
                                 'of time series equals {}.'.format(n_dim))

        X_pad = self.add_est_points(X, t)  # add padding
        sr = self.saliency_map(X_pad)  # compute saliency map
        sr = sr[:-self.n_est_points]  # remove padding again
        ma_sr = np.convolve(sr, self.conv_local, 'same')
        iscore = (sr - ma_sr) / (ma_sr + EPSILON)
        return iscore

    def predict(self,
                X: np.ndarray,
                t: np.ndarray = None,
                return_instance_score: bool = True) \
            -> Dict[Dict[str, str], Dict[np.ndarray, np.ndarray]]:
        """
        Compute outlier scores and transform into outlier predictions.

        Parameters
        ----------
        X
            Time series of instances.
        t
            Time steps.
        return_instance_score
            Whether to return instance level outlier scores.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the outlier predictions and instance level outlier scores.
        """
        if t is None:
            t = np.arange(X.shape[0])

        # compute outlier scores
        iscore = self.score(X.reshape(-1, ), t)

        # values above threshold are outliers
        outlier_pred = (iscore > self.threshold).astype(int)

        # populate output dict
        od = outlier_prediction_dict()
        od['meta'] = self.meta
        od['data']['is_outlier'] = outlier_pred
        if return_instance_score:
            od['data']['instance_score'] = iscore
        return od


class MySpectralResidual(object):

    def __init__(self, threshold: float = None, window_amp: int = None,
                 window_local: int = None, n_est_points: int = None, n_grad_points: int = 5, t: int = 1) -> None:

        self.sr = SpectralResidual(threshold, window_amp, window_local, n_est_points, n_grad_points)
        self.t = np.arange(t)

    @staticmethod
    def __calc_batch__(input_numpy: np.ndarray, sr: SpectralResidual) -> np.ndarray:
        rtn = sr.predict(input_numpy, return_instance_score=True)['data']['instance_score']
        return np.reshape(rtn, newshape=(1, -1))

    def __call__(self, input: np.ndarray) -> np.ndarray:
        ''' from memory key to memory value
        Args:
            input (np.ndarray): the input data points, [batch_size, memory_size]
        '''
        batch_size = input.shape[0]
        input_numpy = np.reshape(input, newshape=(batch_size, -1, 1))
        rtns = Parallel(n_jobs=-1)(
            delayed(MySpectralResidual.__calc_batch__)(input_numpy[idx, :], self.sr)
            for idx in range(batch_size)
        )
        outputs = np.concatenate(rtns, axis=0)
        return outputs
