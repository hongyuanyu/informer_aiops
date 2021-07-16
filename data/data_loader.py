import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.andet import kde, sr
from utils.tools import StandardScaler, padding
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]

        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]
        
        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin+self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_AIOPS(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='/home/hongyuan/ali/Informer2020/data/aiops/', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 32*4
            self.label_len = 32
            self.pred_len = 32
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init

        assert flag in ['train', 'val', 'test'], 'The mode [{0}] not implemented'.format(flag)
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.no_ssl = False
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        train_file_name = 'phase2_train.csv'
        test_file_name = 'phase2_ground_truth.hdf'
        if self.set_type == 0:
            df = pd.read_csv(os.path.join(self.data_path, train_file_name))
        else:
            df = pd.read_hdf(os.path.join(self.data_path, test_file_name))
       

        if self.features=='kde':
            cfgs = {
                'kernel': 'gaussian',
                'bandwidth': 0.2
            }
            feature_model = kde.KernelDensity(**cfgs)
        elif self.features=='sr':
	    # configurations for Spectral Residual
            cfgs = {
                'threshold': 0.9,
                'window_amp': 20,
                'window_local': 20,
                'n_est_points': 10,
                'n_grad_points': 5,
                't': 1
            }
            feature_model = sr.MySpectralResidual(**cfgs)
        else:
            feature_model = None


        data_points, data_stamps, labels, dataset_lens = [], [], [], []

        kpi_name = None
        # converting KPI ID type
        df['KPI ID'] = df['KPI ID'].astype(str)
        if kpi_name is not None:
            rows = pd.DataFrame.copy(df[df['KPI ID'] == kpi_name])

            # sorting for correcting timestamp
            rows.sort_values('timestamp', ascending=True,  inplace=True)
            dataset = rows.iloc[:, [0, 1, 2]]
            dataset_numpy = np.array(dataset)
            timestamp, point, label = dataset_numpy[:, 0], dataset_numpy[:, 1], dataset_numpy[:, 2]

            data_stamps.append(timestamp)
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
                dataset = rows.iloc[:, [0, 1, 2]]

                dataset_numpy = np.array(dataset)
                if dataset_numpy.shape[0] == 0:
                    continue

                timestamp, point, label = dataset_numpy[:, 0], dataset_numpy[:, 1], dataset_numpy[:, 2]
                begin += point.shape[0]

                data_stamps.append(timestamp)
                data_points.append(point)
                labels.append(label)
                dataset_lens.append(begin)

        self.data_x = []
        if self.scale:
            for data_point in data_points:
                self.scaler.fit(data_point)
                self.data_x.append(self.scaler.transform(data_point))
        else:
            self.data_x = data_points

        self.data_stamp = []
        for timestamp in data_stamps:
            timestamp = pd.to_datetime(timestamp, unit='s')
            data_stamp = time_features(timestamp, timeenc=self.timeenc, freq=self.freq)
            self.data_stamp.append(data_stamp)
            
        if self.inverse:
            self.data_y = data_points
        else:
            self.data_y = self.data_x

        if self.no_ssl:
            self.data_y = labels

        self.seq_lens = [(x.shape[0] - self.seq_len - self.pred_len) for x in self.data_x]
        self.seq_lens_sum = [sum(self.seq_lens[:i+1]) for i in range(len(self.seq_lens))]
        self.seq_lens_sum.insert(0, 0)
    
    def __getitem__(self, index):
        raw_index = index
        flag = (raw_index - np.array(self.seq_lens_sum)) >= 0
        seq_idx = sum(flag) - 1
        index = raw_index - self.seq_lens_sum[seq_idx]

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[seq_idx][s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[seq_idx][r_begin:r_begin+self.label_len], self.data_y[seq_idx][r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[seq_idx][r_begin:r_end]
        seq_x_mark = self.data_stamp[seq_idx][s_begin:s_end]
        seq_y_mark = self.data_stamp[seq_idx][r_begin:r_end]

        if seq_x.shape[0] < 96 or seq_y.shape[0] < 72:
            import pdb; pdb.set_trace()

        return np.expand_dims(seq_x, axis=-1), np.expand_dims(seq_y, axis=-1), seq_x_mark, seq_y_mark
    
    def __len__(self):
        return self.seq_lens_sum[-1]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_AIOPS_C(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='/home/hongyuan/ali/Informer2020/data/aiops/', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 32*4
            self.label_len = 32
            self.pred_len = 32
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init

        assert flag in ['train', 'val', 'test'], 'The mode [{0}] not implemented'.format(flag)
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.no_ssl = True
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        train_file_name = 'phase2_train.csv'
        test_file_name = 'phase2_ground_truth.hdf'
        if self.set_type == 0:
            df = pd.read_csv(os.path.join(self.data_path, train_file_name))
        else:
            df = pd.read_hdf(os.path.join(self.data_path, test_file_name))
       

        if self.features=='kde':
            cfgs = {
                'kernel': 'gaussian',
                'bandwidth': 0.2
            }
            feature_model = kde.KernelDensity(**cfgs)
        elif self.features=='sr':
	    # configurations for Spectral Residual
            cfgs = {
                'threshold': 0.9,
                'window_amp': 20,
                'window_local': 20,
                'n_est_points': 10,
                'n_grad_points': 5,
                't': 1
            }
            feature_model = sr.MySpectralResidual(**cfgs)
        else:
            feature_model = None


        data_points, data_stamps, labels, dataset_lens = [], [], [], []

        kpi_name = None
        # converting KPI ID type
        df['KPI ID'] = df['KPI ID'].astype(str)
        if kpi_name is not None:
            rows = pd.DataFrame.copy(df[df['KPI ID'] == kpi_name])

            # sorting for correcting timestamp
            rows.sort_values('timestamp', ascending=True,  inplace=True)
            dataset = rows.iloc[:, [0, 1, 2]]
            dataset_numpy = np.array(dataset)
            timestamp, point, label = dataset_numpy[:, 0], dataset_numpy[:, 1], dataset_numpy[:, 2]

            data_stamps.append(timestamp)
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
                dataset = rows.iloc[:, [0, 1, 2]]

                dataset_numpy = np.array(dataset)
                if dataset_numpy.shape[0] == 0:
                    continue

                timestamp, point, label = dataset_numpy[:, 0], dataset_numpy[:, 1], dataset_numpy[:, 2]
                begin += point.shape[0]

                data_stamps.append(timestamp)
                data_points.append(point)
                labels.append(label)
                dataset_lens.append(begin)

        self.data_x = []
        if self.scale:
            for data_point in data_points:
                self.scaler.fit(data_point)
                self.data_x.append(self.scaler.transform(data_point))
        else:
            self.data_x = data_points

        self.data_stamp = []
        for timestamp in data_stamps:
            timestamp = pd.to_datetime(timestamp, unit='s')
            data_stamp = time_features(timestamp, timeenc=self.timeenc, freq=self.freq)
            self.data_stamp.append(data_stamp)
            

        self.data_y = labels

        self.seq_lens = [(x.shape[0]) for x in self.data_x]
        self.seq_lens_sum = [sum(self.seq_lens[:i+1]) for i in range(len(self.seq_lens))]
        self.seq_lens_sum.insert(0, 0)
    
    def __getitem__(self, index):
        raw_index = index
        flag = (raw_index - np.array(self.seq_lens_sum)) >= 0
        seq_idx = sum(flag) - 1
        index = raw_index - self.seq_lens_sum[seq_idx]

        
        #s_begin = index
        #s_end = s_begin + self.seq_len
        #r_begin = s_end - self.label_len
        #r_end = r_begin + self.label_len + self.pred_len
        
        s_end = index
        s_begin = s_end - self.seq_len        
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        if s_begin < 0:
            seq_x = padding(self.data_x[seq_idx][0:s_end+1], self.seq_len, np.float32, None, False)
            seq_x_mark = padding(self.data_stamp[seq_idx][0:s_end+1], self.seq_len, np.float32, None, False)
        else:
            seq_x = self.data_x[seq_idx][s_begin+1:s_end+1]
            seq_x_mark = self.data_stamp[seq_idx][s_begin+1:s_end+1]

        if r_begin < 0:
            seq_y_mark = padding(self.data_stamp[seq_idx][0:r_end+1], self.label_len + self.pred_len, np.float32, None, False)
        elif r_end >= self.data_stamp[seq_idx].shape[0]:
            seq_y_mark = padding(self.data_stamp[seq_idx][r_begin+1::][::-1].copy(), self.label_len + self.pred_len, np.float32, None, False)
            seq_y_mark = seq_y_mark[::-1].copy()
        else:
            seq_y_mark = self.data_stamp[seq_idx][r_begin+1:r_end+1]

        label_y = self.data_y[seq_idx][index]
        if label_y:
            class_y = seq_idx * 2 + 1
        else:
            class_y = seq_idx * 2 

        if seq_x.shape[0] < 96 or seq_y_mark.shape[0] < 72 or seq_x_mark.shape[0] < 96:
            import pdb; pdb.set_trace()
         
        return np.expand_dims(seq_x, axis=-1), seq_x_mark, seq_y_mark, class_y
    
    def __len__(self):
        return self.seq_lens_sum[-1]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


