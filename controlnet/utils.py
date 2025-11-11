import numpy as np
import pandas as pd
import os
from pandas.api.types import is_numeric_dtype
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
from scipy.interpolate import PchipInterpolator
import torch.nn.functional as F
import random
import torch
import json  
import pickle
import math

def load_json(path):
	with open(path, "r") as f:
		data = json.load(f)
	return data 

def save_pickle(path, data):
	with open(path, "wb") as f:
		pickle.dump(data, f)
		

def tensor_add_laplace_noise(tensor, scale):
    noise = torch.tensor(np.random.laplace(0, scale, tensor.size()), dtype=tensor.dtype)
    noisy_tensor = tensor + noise
    return noisy_tensor

def tensor_add_uniform_noise(tensor, scale):
    noise = torch.empty_like(tensor).uniform_(-scale, scale)
    noisy_tensor = tensor + noise
    return noisy_tensor

def tensor_add_normal_noise(tensor, scale):
    std_dev = scale
    noise = torch.randn_like(tensor) * std_dev
    return tensor + noise

def array_add_laplace_noise(arr: np.ndarray, scale: float) -> np.ndarray:
    noise = np.random.laplace(loc=0.0, scale=scale, size=arr.shape)
    return arr + noise


def array_add_uniform_noise(arr: np.ndarray, scale: float) -> np.ndarray:
    noise = np.random.uniform(low=-scale, high=scale, size=arr.shape)
    return arr + noise


def array_add_normal_noise(arr: np.ndarray, std: float) -> np.ndarray:
    noise = np.random.normal(loc=0.0, scale=std, size=arr.shape)
    return arr + noise

def prepare_fast_dataloader(
    D,
    shuffle: bool,
    batch_size: int,
	moretensors=None
):
    dataloader = FastTensorDataLoader(D, batch_size=batch_size, shuffle=shuffle, moretensors=moretensors)
    while True:
        yield from dataloader

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, tensors, batch_size=32, shuffle=False, moretensors=None):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.n_dim = tensors[0].shape[0]
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.moretensors = moretensors
        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
            if self.moretensors is not None:
                self.moretensors = [t[r] for t in self.moretensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        if self.moretensors is not None:
            morebatch = tuple(t[self.i:self.i+self.batch_size] for t in self.moretensors)
            return batch+morebatch
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

def data_preprocessing(raw_data, label, save_dir=None):
    data_wrapper = DataWrapper()
    label_wrapper = DataWrapper()
    data_wrapper.fit(raw_data)
    label_wrapper.fit(raw_data[[label]])

    if save_dir is not None:
        save_pickle(data=data_wrapper, path=os.path.join(f'tabsyn/ckpt/{save_dir}', 'data_wrapper.pkl'))
        save_pickle(data=label_wrapper, path=os.path.join(f'tabsyn/ckpt/{save_dir}', 'label_wrapper.pkl'))
    return data_wrapper, label_wrapper

class DataWrapper:
	def __init__(self, num_encoder="quantile", seed=0):
		self.num_encoder = num_encoder
		self.seed = 0
		self.word_freq = []

	def cal_frec(self, df):
		idx = 0
		nums = df.shape[0]
		for col in df.columns:
			self.word_freq.append({})
			words = df[col]
			freq = words.value_counts()
			for key in freq.keys():
				self.word_freq[idx][key] = freq[key] / nums
			idx += 1
		return
	
	def transto_frec(self,data):
		num_cols = data.shape[1]
		for col in range(num_cols): #ith column
			col_data = data[:, col]
			for idx in range(col_data.shape[0]): #jth data in ith column
				if isinstance(col_data[idx], (int,float)) and math.isnan(float(col_data[idx])):
					print('nan occurred')
					col_data[idx] = 0
				else:
					col_data[idx] = self.word_freq[col][col_data[idx]]
		return data.astype(float)

	def fit(self, dataframe, all_category = False):
		self.raw_dim = dataframe.shape[1]
		self.raw_columns = dataframe.columns
		self.all_distinct_values = {}	# For categorical columns
		self.num_normalizer = {}   # For numerical columns
		self.num_dim = 0
		self.columns = []
		self.col_dim = []
		self.col_dtype = {}	
		for i, col in enumerate(self.raw_columns):
			if all_category:
				break
			if is_numeric_dtype(dataframe[col]) and dataframe[col].dtype != 'bool':
				col_data = dataframe.loc[pd.notna(dataframe[col])][col]
				self.col_dtype[col] = col_data.dtype
				if self.num_encoder == "quantile":
					self.num_normalizer[col] = QuantileTransformer(
												output_distribution='normal',
												n_quantiles=max(min(len(col_data) // 30, 1000), 10),
												subsample=1000000000,
												random_state=self.seed,)
				elif self.num_encoder == "standard":
					self.num_normalizer[col] = StandardScaler()
				elif self.num_encoder == "minmax":
					self.num_normalizer[col] = MinMaxScaler()
				else:
					raise ValueError(f"Unknown num encoder: {self.num_encoder}")
				self.num_normalizer[col].fit(col_data.values.reshape(-1, 1))
				self.columns.append(col)
				self.num_dim += 1
				self.col_dim.append(1)
		for i, col in enumerate(self.raw_columns):
			if col not in self.num_normalizer.keys():
				col_data = dataframe.loc[pd.notna(dataframe[col])][col]
				self.col_dtype[col] = col_data.dtype
				distinct_values = col_data.unique()
				distinct_values.sort()
				self.all_distinct_values[col] = distinct_values
				self.columns.append(col)
				self.col_dim.append(max(1, int(np.ceil(np.log2(len(distinct_values))))))

	def transform(self, data):
		# normalize the numreical column and transform the categorical data to oridinal type
		reorder_data = data[self.columns].values
		norm_data = []
		for i, col in enumerate(self.columns):
			col_data = reorder_data[:, i]
			if col in self.all_distinct_values.keys():
				col_data = self.CatValsToNum(col, col_data).reshape(-1, 1)
				col_data = self.ValsToBit(col_data, self.col_dim[i])
				norm_data.append(col_data)
			elif col in self.num_normalizer.keys():
				norm_data.append(self.num_normalizer[col].transform(col_data.reshape(-1, 1)).reshape(-1, 1))
		norm_data = np.concatenate(norm_data, axis=1)
		norm_data = norm_data.astype(np.float32)
		return norm_data

	def ReOrderColumns(self, data: pd.DataFrame):
		ndf = pd.DataFrame([])
		for col in self.raw_columns:
			ndf[col] = data[col]
		return ndf

	def GetColData(self, data, col_id):
		col_index = np.cumsum(self.col_dim)
		col_data = data.copy()
		if col_id == 0:
			return col_data[:, :col_index[0]]
		else:
			return col_data[:, col_index[col_id-1]:col_index[col_id]]

	def ValsToBit(self, values, bits):
		bit_values = np.zeros((values.shape[0], bits))
		for i in range(values.shape[0]):
			bit_val = np.mod(np.right_shift(int(values[i]), list(reversed(np.arange(bits)))), 2)
			bit_values[i, :] = bit_val
		return bit_values

	def BitsToVals(self, bit_values):
		bits = bit_values.shape[1]
		values = bit_values.astype(int)
		values = values * (2 ** np.array(list((reversed(np.arange(bits))))))
		values = np.sum(values, axis=1)
		return values

	def CatValsToNum(self, col, values):	
		num_values = pd.Categorical(values, categories=self.all_distinct_values[col]).codes
		# num_values = np.zeros_like(values)
		# for i, val in enumerate(values):
		# 	ind = np.where(self.all_distinct_values[col] == val)
		# 	num_values[i] = ind[0][0]
		return num_values

	def NumValsToCat(self, col, values):
		cat_values = np.zeros_like(values).astype(object)
		#print(col_name, values)
		values = np.clip(values, 0, len(self.all_distinct_values[col])-1)
		for i, val in enumerate(values):
			#val = np.clip(val, self.Mins[col_id], self.Maxs[col_id])
			cat_values[i] = self.all_distinct_values[col][int(val)]
		return cat_values 

	def ReverseToOrdi(self, data):
		reverse_data = []
		
		# Unnorm the normalized numerical columns, and reverse the binary code to ordinal columns
		for i, col in enumerate(self.columns):
			#print(col_name)
			col_data = self.GetColData(data, i)
			if col in self.all_distinct_values.keys():
				col_data = np.round(col_data)
				col_data = self.BitsToVals(col_data)
				col_data = col_data.astype(np.int32)
			else:
				col_data = self.num_normalizer[col].inverse_transform(col_data.reshape(-1, 1)) 
				if self.col_dtype[col] == np.int32 or self.col_dtype[col] == np.int64:
					col_data = np.round(col_data).astype(int)
				#col_data = self.NumValsToCat(col, col_data)
			#col_data = col_data.astype(self.raw_data[col].dtype)
			reverse_data.append(col_data.reshape(-1, 1))
		reverse_data = np.concatenate(reverse_data, axis=1)
		return reverse_data

	def ReverseToCat(self, data):
		reverse_data = []
		for i, col in enumerate(self.columns):
			col_data = data[:, i]
			if col in self.all_distinct_values.keys():
				col_data = self.NumValsToCat(col, col_data)
			reverse_data.append(col_data.reshape(-1, 1))
		reverse_data = np.concatenate(reverse_data, axis=1)
		return reverse_data		

	def Reverse(self, data):
		data = self.ReverseToOrdi(data)
		data = self.ReverseToCat(data)
		data = pd.DataFrame(data, columns=self.columns)
		return self.ReOrderColumns(data)

	def RejectSample(self, sample):
		all_index = set(range(sample.shape[0]))
		allow_index = set(range(sample.shape[0]))
		for i, col in enumerate(self.columns):
			if col in self.all_distinct_values.keys():
				allow_index = allow_index & set(np.where(sample[:, i]<len(self.all_distinct_values[col]))[0])
				allow_index = allow_index & set(np.where(sample[:, i]>=0)[0])
		reject_index = all_index - allow_index
		allow_index = np.array(list(allow_index))
		reject_index = np.array(list(reject_index))
		#allow_sample = sample[allow_index, :]
		return allow_index, reject_index