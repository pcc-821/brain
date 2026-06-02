"""
MNIST数据集处理和预处理
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_mnist_sklearn():
    """
    使用sklearn加载MNIST数据集（8x8数字）
    
    返回值:
    --------
    tuple
        (X_train, y_train, X_test, y_test)
    """
    # 加载数字数据集（0-9的8x8图像）
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # 归一化到[0, 1]
    X = X / 16.0
    
    # 分割为训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, y_train, X_test, y_test


def load_mnist_full():
    """
    加载完整的MNIST数据集（28x28图像）
    如果本地不可用，则从互联网下载
    
    返回值:
    --------
    tuple
        (X_train, y_train, X_test, y_test)
    """
    try:
        from tensorflow import keras
        # 加载完整的MNIST数据集
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        
        # 展平并归一化
        X_train = X_train.reshape(-1, 28*28) / 255.0
        X_test = X_test.reshape(-1, 28*28) / 255.0
        
        return X_train, y_train, X_test, y_test
    
    except ImportError:
        print("TensorFlow不可用。使用sklearn数字数据集代替。")
        return load_mnist_sklearn()


def convert_to_spike_train(image, threshold=0.5, duration=100, dt=1.0):
    """
    使用阈值方法将图像转换为脉冲列车
    
    参数:
    -----------
    image : array
        输入图像（展平）
    threshold : float
        脉冲发放的阈值
    duration : int
        脉冲列车的持续时间（时间步数）
    dt : float
        时间步长（毫秒）
        
    返回值:
    --------
    array
        形状为(神经元数, 时间步数)的脉冲列车
    """
    num_neurons = len(image)
    spike_train = np.zeros((num_neurons, int(duration / dt)))
    
    # 发放率与像素强度成正比
    for i, pixel_value in enumerate(image):
        if pixel_value > 0:
            # 发放率（赫兹）
            firing_rate = pixel_value * 100  # 最大100赫兹
            # 脉冲数量
            num_spikes = max(1, int(firing_rate * duration / 1000))
            # 随机脉冲时间
            spike_times = np.random.choice(
                int(duration / dt), size=num_spikes, replace=False
            )
            spike_train[i, spike_times] = 1
    
    return spike_train


def convert_to_poisson_spike_train(image, duration=100, dt=1.0):
    """
    将图像转换为泊松脉冲列车
    
    参数:
    -----------
    image : array
        输入图像（展平）
    duration : int
        脉冲列车的持续时间（时间步数）
    dt : float
        时间步长（毫秒）
        
    返回值:
    --------
    array
        形状为(神经元数, 时间步数)的脉冲列车
    """
    num_neurons = len(image)
    num_steps = int(duration / dt)
    spike_train = np.zeros((num_neurons, num_steps))
    
    # 生成泊松脉冲列车
    for i, pixel_value in enumerate(image):
        if pixel_value > 0:
            # 发放率（赫兹）
            firing_rate = pixel_value * 100  # 最大100赫兹
            # 每个时间步的脉冲概率
            spike_prob = firing_rate * dt / 1000
            spike_train[i, :] = np.random.binomial(1, spike_prob, num_steps)
    
    return spike_train


class SpikeDataset:
    """
    带有脉冲列车转换的数据集
    
    参数:
    -----------
    X : array
        输入数据
    y : array
        标签
    duration : int
        脉冲列车的持续时间（毫秒）
    dt : float
        时间步长（毫秒）
    spike_encoding : str
        编码方法：'threshold'或'poisson'
    """
    
    def __init__(self, X, y, duration=100, dt=1.0, spike_encoding='poisson'):
        self.X = X
        self.y = y
        self.duration = duration
        self.dt = dt
        self.spike_encoding = spike_encoding
        self.num_steps = int(duration / dt)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        """获取样本的脉冲列车和标签"""
        image = self.X[idx]
        label = self.y[idx]
        
        if self.spike_encoding == 'threshold':
            spike_train = convert_to_spike_train(image, duration=self.duration, dt=self.dt)
        elif self.spike_encoding == 'poisson':
            spike_train = convert_to_poisson_spike_train(image, duration=self.duration, dt=self.dt)
        else:
            raise ValueError(f"未知的脉冲编码方法: {self.spike_encoding}")
        
        return spike_train, label
    
    def get_batch(self, indices, precompute=False):
        """
        获取脉冲列车批次
        
        参数:
        -----------
        indices : array-like
            要检索的索引
        precompute : bool
            是否预先计算所有脉冲列车
            
        返回值:
        --------
        tuple
            (spike_trains, labels)，其中spike_trains的形状为(批次大小, 神经元数, 时间步数)
        """
        spike_trains = []
        labels = []
        
        for idx in indices:
            spike_train, label = self[idx]
            spike_trains.append(spike_train)
            labels.append(label)
        
        return np.array(spike_trains), np.array(labels)


def preprocess_data(X, normalization='minmax'):
    """
    预处理输入数据
    
    参数:
    -----------
    X : array
        输入数据
    normalization : str
        归一化方法：'minmax'、'standard'或'none'
        
    返回值:
    --------
    array
        预处理后的数据
    """
    if normalization == 'minmax':
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_processed = (X - X_min) / (X_max - X_min + 1e-8)
    
    elif normalization == 'standard':
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X)
    
    elif normalization == 'none':
        X_processed = X
    
    else:
        raise ValueError(f"未知的归一化方法: {normalization}")
    
    return X_processed
