"""
MNIST dataset processing and preprocessing
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_mnist_sklearn():
    """
    Load MNIST dataset using sklearn (8x8 digits)
    
    Returns:
    --------
    tuple
        (X_train, y_train, X_test, y_test)
    """
    # Load digit dataset (8x8 images of digits 0-9)
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # Normalize to [0, 1]
    X = X / 16.0
    
    # Split into training/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, y_train, X_test, y_test


def load_mnist_full():
    """
    Load full MNIST dataset (28x28 images)
    Downloaded from internet if not available locally
    
    Returns:
    --------
    tuple
        (X_train, y_train, X_test, y_test)
    """
    try:
        from tensorflow import keras
        # Load full MNIST dataset
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        
        # Flatten and normalize
        X_train = X_train.reshape(-1, 28*28) / 255.0
        X_test = X_test.reshape(-1, 28*28) / 255.0
        
        return X_train, y_train, X_test, y_test
    
    except ImportError:
        print("TensorFlow not available. Using sklearn digit dataset instead.")
        return load_mnist_sklearn()


def convert_to_spike_train(image, threshold=0.5, duration=100, dt=1.0):
    """
    Convert image to spike train using threshold method
    
    Parameters:
    -----------
    image : array
        Input image (flattened)
    threshold : float
        Threshold for spike firing
    duration : int
        Duration of spike train (timesteps)
    dt : float
        Time step length (milliseconds)
        
    Returns:
    --------
    array
        Spike train with shape (num_neurons, num_timesteps)
    """
    num_neurons = len(image)
    spike_train = np.zeros((num_neurons, int(duration / dt)))
    
    # Firing rate proportional to pixel intensity
    for i, pixel_value in enumerate(image):
        if pixel_value > 0:
            # Firing rate (Hz)
            firing_rate = pixel_value * 100  # Max 100 Hz
            # Number of spikes
            num_spikes = max(1, int(firing_rate * duration / 1000))
            # Random spike times
            spike_times = np.random.choice(
                int(duration / dt), size=num_spikes, replace=False
            )
            spike_train[i, spike_times] = 1
    
    return spike_train


def convert_to_poisson_spike_train(image, duration=100, dt=1.0):
    """
    Convert image to Poisson spike train
    
    Parameters:
    -----------
    image : array
        Input image (flattened)
    duration : int
        Duration of spike train (timesteps)
    dt : float
        Time step length (milliseconds)
        
    Returns:
    --------
    array
        Spike train with shape (num_neurons, num_timesteps)
    """
    num_neurons = len(image)
    num_steps = int(duration / dt)
    spike_train = np.zeros((num_neurons, num_steps))
    
    # Generate Poisson spike train
    for i, pixel_value in enumerate(image):
        if pixel_value > 0:
            # Firing rate (Hz)
            firing_rate = pixel_value * 100  # Max 100 Hz
            # Spike probability for each timestep
            spike_prob = firing_rate * dt / 1000
            spike_train[i, :] = np.random.binomial(1, spike_prob, num_steps)
    
    return spike_train


class SpikeDataset:
    """
    Dataset with spike train conversion
    
    Parameters:
    -----------
    X : array
        Input data
    y : array
        Labels
    duration : int
        Duration of spike train (milliseconds)
    dt : float
        Time step length (milliseconds)
    spike_encoding : str
        Encoding method: 'threshold' or 'poisson'
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
        """Get spike train and label for a sample"""
        image = self.X[idx]
        label = self.y[idx]
        
        if self.spike_encoding == 'threshold':
            spike_train = convert_to_spike_train(image, duration=self.duration, dt=self.dt)
        elif self.spike_encoding == 'poisson':
            spike_train = convert_to_poisson_spike_train(image, duration=self.duration, dt=self.dt)
        else:
            raise ValueError(f"Unknown spike encoding method: {self.spike_encoding}")
        
        return spike_train, label
    
    def get_batch(self, indices, precompute=False):
        """
        Get batch of spike trains
        
        Parameters:
        -----------
        indices : array-like
            Indices to retrieve
        precompute : bool
            Whether to precompute all spike trains
            
        Returns:
        --------
        tuple
            (spike_trains, labels) where spike_trains has shape (batch_size, num_neurons, num_timesteps)
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
    Preprocess input data
    
    Parameters:
    -----------
    X : array
        Input data
    normalization : str
        Normalization method: 'minmax', 'standard', or 'none'
        
    Returns:
    --------
    array
        Preprocessed data
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
        raise ValueError(f"Unknown normalization method: {normalization}")
    
    return X_processed
