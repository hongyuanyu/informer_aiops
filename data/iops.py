import os
import time
import torch
import bisect
import requests
import numpy as np
import pandas as pd
from .andet import kde, sr
from typing import List, Dict
import torch.utils.data as data
from sklearn.cluster import KMeans
from .transforms import paddingZeros
from typing import Callable, Optional, Tuple


def downloader(file_url: str, file_path: str) -> None:
    ''' Downloading the file
    '''
    dir_path, file_name = os.path.split(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file = requests.get(file_url, stream=True)
    with open(file_path, 'wb+') as f:
        for chunk in file.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    print('The file [{0}] downloaded'.format(file_name))


class IOPS(data.Dataset):
    ''' IOPS Contest dataset
        For speeding up the loader, we move the dataset to GPu firstly.
    '''

    train_file_name = 'phase2_train.csv'
    test_file_name = 'phase2_ground_truth.hdf'

    #train_file_url = 'https://antsys-rocktimeseries.cn-hangzhou-alipay-b.oss-cdn.aliyun-inc.com/qiaosu/data/phase2_train.csv'
    #test_file_url = 'https://antsys-rocktimeseries.cn-hangzhou-alipay-b.oss-cdn.aliyun-inc.com/qiaosu/data/phase2_ground_truth_new.csv'

    base_folder = 'iops-dataset'

    names = ['timestamp', 'value', 'label', 'KPI ID']

    def __init__(self, root_dir, device: torch.device = None,
                 value_models: List = ['kde'], cfgs: Dict = None,
                 kpi_name=None, win_size: int = 32, memory_size: int = 128,
                 mode: str = 'train', download: bool = False, transform: Optional[Callable] = None, with_cluster: bool = False) -> None:
        super(IOPS, self).__init__()
        assert mode in ['train', 'test'], 'The mode [{0}] not implemented'.format(mode)

        self.root_dir = root_dir
        self.win_size = win_size
        self.transform = transform
        self.memory_size = memory_size

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        if download:
            # downloading train fille
            downloader(self.train_file_url, os.path.join(self.root_dir, self.base_folder, self.train_file_name))
            # downloading test file
            downloader(self.test_file_url, os.path.join(self.root_dir, self.base_folder, self.test_file_name))

        self.root_dir = os.path.join(self.root_dir, self.base_folder)
        if mode == 'train':
            df = pd.read_csv(os.path.join(self.root_dir, self.train_file_name))
        else:
            df = pd.read_hdf(os.path.join(self.root_dir, self.test_file_name))

        self.value_models = []
        if 'kde' in value_models:
            _kde = kde.KernelDensity(**cfgs['kde'])
            self.value_models.append(_kde)

        if 'sr' in value_models:
            _sr = sr.MySpectralResidual(**cfgs['sr'])
            self.value_models.append(_sr)

        data_points, labels, dataset_lens = [], [], []
        # converting KPI ID type
        df['KPI ID'] = df['KPI ID'].astype(str)
        if kpi_name is not None:
            rows = pd.DataFrame.copy(df[df['KPI ID'] == kpi_name])

            # sorting for correcting timestamp
            rows.sort_values('timestamp', ascending=True,  inplace=True)
            dataset = rows.iloc[:, [1, 2]]
            dataset_numpy = np.array(dataset)
            point, label = dataset_numpy[:, 0], dataset_numpy[:, 1]

            data_points.append(point)
            labels.append(label)
            dataset_lens.append(labels[-1].shape[0])
        else:
            kpi_names = dict(df['KPI ID'].value_counts()).keys()
            kpi_names = sorted(kpi_names)

            begin = 0
            for kpi_name in kpi_names:
                rows = pd.DataFrame.copy(df[df['KPI ID'] == kpi_name])

                # sorting for correcting timestamp
                rows.sort_values('timestamp', ascending=True,  inplace=True)
                dataset = rows.iloc[:, [1, 2]]

                dataset_numpy = np.array(dataset)
                if dataset_numpy.shape[0] == 0:
                    continue

                point, label = dataset_numpy[:, 0], dataset_numpy[:, 1]
                begin += point.shape[0]

                data_points.append(point)
                labels.append(label)
                dataset_lens.append(begin)

        if with_cluster is True:
            self.memory_size = self.memory_size // 2

        # concatenate multiple kpis or one kpi
        self.labels = np.concatenate(labels, axis=0).astype(np.long)
        print('---------------- Fetch {} dataset begining ----------------'.format(mode))
        start_time = time.time()
        self.wdw_points, self.mry_key_points = self.__prefetcher__(data_points, dataset_lens)
        self.mry_value_points = self.value_models[0](self.mry_key_points)
        end_time = time.time()
        print('---------------- Fetch {} dataset finished [{:.2f} s]----------------'.format(mode, end_time - start_time))

        if with_cluster is True:
            print('---------------- Clustering {} dataset begining ----------------'.format(mode))
            start_time = time.time()
            begin = 0
            clr_key_points_list, clr_value_points_list = [], []
            for end in dataset_lens:
                dataset = self.wdw_points[begin: end, :]
                # clustering configuration
                kmeans = KMeans(n_clusters=self.memory_size // self.win_size, init='k-means++', n_init=10)
                kmeans.fit(dataset)
                cluster_centers = kmeans.cluster_centers_
                cluster_centers_values = self.value_models[0](cluster_centers)

                clr_centers = cluster_centers.reshape(-1, self.memory_size)
                clr_centers_values = cluster_centers_values.reshape(-1, self.memory_size)
                clr_key_points = np.repeat(clr_centers, repeats=end - begin, axis=0)
                clr_value_points = np.repeat(clr_centers_values, repeats=end - begin, axis=0)

                clr_key_points_list.append(clr_key_points)
                clr_value_points_list.append(clr_value_points)
                begin = end

            clr_key_points_list = np.concatenate(clr_key_points_list, axis=0)
            clr_value_points_list = np.concatenate(clr_value_points_list, axis=0)
            self.mry_key_points = np.concatenate([self.mry_key_points, clr_key_points_list], axis=1)
            self.mry_value_points = np.concatenate([self.mry_value_points, clr_value_points_list], axis=1)
            end_time = time.time()
            print('---------------- Clustering {} dataset finished [{:.2f} s]----------------'.format(mode, end_time - start_time))
        # save memory
        np.savez(os.path.join(self.root_dir, 'tmp-{}-iops.npz'.format(mode)),
                 wdw_points=self.wdw_points, mry_key_points=self.mry_key_points,
                 mry_value_points=self.mry_value_points, labels=self.labels)
        # dtype setting
        self.labels = self.labels.astype(np.long)
        self.wdw_points = self.wdw_points.astype(np.float32)
        self.mry_key_points = self.mry_key_points.astype(np.float32)
        self.mry_value_points = self.mry_value_points.astype(np.float32)

        np.savez(os.path.join(self.root_dir, '{}-iops.npz'.format(mode)),
                 wdw_points=self.wdw_points, mry_key_points=self.mry_key_points,
                 mry_value_points=self.mry_value_points, labels=self.labels)

        if device is not None:
            self.labels = torch.as_tensor(self.labels, dtype=torch.long, device=device)
            self.wdw_points = torch.as_tensor(self.wdw_points, dtype=torch.float32, device=device)
            self.mry_key_points = torch.as_tensor(self.mry_key_points, dtype=torch.float32, device=device)
            self.mry_value_points = torch.as_tensor(self.mry_value_points, dtype=torch.float32, device=device)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __prefetcher__(self, data_points: List[np.ndarray], dataset_lens: List[int]) -> Tuple[np.ndarray]:
        ''' get the window size and memory size of data points
        '''
        wdw_points_list, mry_points_list = [], []
        for index in range(dataset_lens[-1]):
            dataset_idx = bisect.bisect_left(dataset_lens, index + 1)
            if dataset_idx > 0:
                index = index - dataset_lens[dataset_idx - 1]

            wdw_points, points = None, None
            # getting the window size of data points
            if index - self.win_size > -1:
                points = data_points[dataset_idx][index - self.win_size: index]
            else:
                points = data_points[dataset_idx][0: index]

            if points is not None:
                wdw_points = points

            mry_points, points = None, None
            # getting the memory size of data points
            if index - self.memory_size - self.win_size > -1:
                points = data_points[dataset_idx][index - self.memory_size - self.win_size: index - self.win_size]
            elif index - self.win_size > -1:
                points = data_points[dataset_idx][0: index - self.win_size]

            if points is not None:
                mry_points = points

            # padding zeros for values, memorys
            wdw_points = paddingZeros(wdw_points, self.win_size, np.float32, None, False)
            mry_points = paddingZeros(mry_points, self.memory_size, np.float32, None, False)
            wdw_points_list.append(np.reshape(wdw_points, newshape=(1, -1)))
            mry_points_list.append(np.reshape(mry_points, newshape=(1, -1)))

        return np.concatenate(wdw_points_list, axis=0), np.concatenate(mry_points_list, axis=0)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        ''' get dataset in index
        '''
        label = self.labels[index]
        wdw_point = self.wdw_points[index]
        mry_key_point = self.mry_key_points[index]
        mry_value_point = self.mry_value_points[index]

        if self.transform is not None:
            wdw_point = self.transform(wdw_point)
            mry_key_point = self.transform(mry_key_point)
            mry_value_point = self.transform(mry_value_point)
        return wdw_point, mry_key_point, mry_value_point, label
