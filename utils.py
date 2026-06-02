"""
Utilities for visualization and metrics
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_spike_raster(spike_times, title="Spike Raster", figsize=(12, 6)):
    """
    Plot spike raster diagram
    
    Parameters:
    -----------
    spike_times : list of lists
        Spike times for each neuron
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for neuron_idx, spikes in enumerate(spike_times):
        ax.vlines(spikes, neuron_idx - 0.5, neuron_idx + 0.5, colors='black')
    
    ax.set_ylabel('Neuron Index')
    ax.set_xlabel('Time (ms)')
    ax.set_title(title)
    ax.set_ylim(-0.5, len(spike_times) - 0.5)
    
    return fig, ax


def plot_membrane_potential(potentials, neuron_indices=None, figsize=(12, 6)):
    """
    Plot membrane potential over time
    
    Parameters:
    -----------
    potentials : list of arrays
        Membrane potentials for each neuron
    neuron_indices : list
        Indices of neurons to plot (None = all)
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if neuron_indices is None:
        neuron_indices = range(min(10, len(potentials)))
    
    for idx in neuron_indices:
        ax.plot(potentials[idx], label=f'Neuron {idx}', alpha=0.7)
    
    ax.set_ylabel('Membrane Potential (mV)')
    ax.set_xlabel('Time Step')
    ax.set_title('Membrane Potentials')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_weight_matrix(weights, title="Weight Matrix", figsize=(8, 8)):
    """
    Plot synaptic weight matrix
    
    Parameters:
    -----------
    weights : array
        Weight matrix (post x pre)
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(weights, cmap='viridis', aspect='auto')
    ax.set_xlabel('Presynaptic Neuron')
    ax.set_ylabel('Postsynaptic Neuron')
    ax.set_title(title)
    
    plt.colorbar(im, ax=ax, label='Weight')
    
    return fig, ax


def plot_training_history(history, figsize=(12, 4)):
    """
    Plot training history
    
    Parameters:
    -----------
    history : dict
        Training history with keys 'accuracy', 'loss'
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    
    if 'accuracy' in history:
        axes.plot(history['accuracy'], marker='o', label='Accuracy')
    
    if 'loss' in history:
        ax2 = axes.twinx()
        ax2.plot(history['loss'], marker='s', color='orange', label='Loss')
        ax2.set_ylabel('Loss')
    
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Accuracy')
    axes.set_title('Training History')
    axes.grid(True, alpha=0.3)
    axes.legend()
    
    return fig, axes


def plot_confusion_matrix(y_true, y_pred, num_classes=10, figsize=(8, 8)):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    num_classes : int
        Number of classes
    figsize : tuple
        Figure size
    """
    # Create confusion matrix
    cm = np.zeros((num_classes, num_classes))
    for true, pred in zip(y_true, y_pred):
        cm[int(true), int(pred)] += 1
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    
    plt.colorbar(im, ax=ax, label='Count')
    
    return fig, ax


def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics
    
    Parameters:
    -----------
    y_true : array
        True labels
    y_pred : array
        Predicted labels
        
    Returns:
    --------
    dict
        Metrics including accuracy, precision, recall, F1
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    return metrics


def print_metrics(metrics):
    """Print classification metrics"""
    print("\n" + "="*50)
    print("Classification Metrics")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key.capitalize():12s}: {value:.4f}")
    print("="*50 + "\n")
