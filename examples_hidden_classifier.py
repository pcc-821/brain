"""
Examples demonstrating hidden layer based digit recognition
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from dataset import preprocess_data, convert_to_poisson_spike_train
from stdp_network_hidden_classifier import STDPNetworkWithHiddenClassifier


def load_mnist_28x28_small(num_train=300, num_test=100):
    """Load MNIST-like data (28x28 version, small subset)"""
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    
    # Upsample to 28x28
    from scipy.ndimage import zoom
    X_images = X.reshape(-1, 8, 8)
    scale_factor = 28 / 8
    X_upsampled = np.array([
        zoom(img, scale_factor, order=1) for img in X_images
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_upsampled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Use smaller subset
    X_train = X_train[:num_train]
    y_train = y_train[:num_train]
    X_test = X_test[:num_test]
    y_test = y_test[:num_test]
    
    return X_train, y_train, X_test, y_test


def example_1_single_digit_analysis():
    """Example 1: Analyze hidden layer response to a single digit"""
    print("\n" + "="*70)
    print("Example 1: Single Digit Analysis")
    print("="*70)
    
    print("\n[1/3] Loading and preparing data...")
    X_train, y_train, _, _ = load_mnist_28x28_small(num_train=100, num_test=50)
    X_train = preprocess_data(X_train.reshape(len(X_train), -1), normalization='minmax')
    X_train = X_train.reshape(-1, 28, 28)
    
    print("[2/3] Creating and training network...")
    network = STDPNetworkWithHiddenClassifier(
        input_width=28,
        input_height=28,
        hidden_size=100,
        num_classes=10
    )
    
    network.train(X_train, y_train, epochs=3, batch_size=10, duration=100)
    
    print("[3/3] Analyzing single digits...")
    
    # Test on a few samples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for digit in range(10):
        ax = axes[digit]
        
        # Find a test sample for this digit
        digit_samples = X_train[y_train == digit]
        if len(digit_samples) > 0:
            test_sample = digit_samples[0]
            
            # Get hidden layer response
            spike_train = convert_to_poisson_spike_train(
                test_sample.flatten(), duration=100, dt=0.1
            )
            hidden_spikes = network.forward(spike_train, duration=100)
            spike_counts = np.array([len(spikes) for spikes in hidden_spikes])
            
            # Visualize
            ax.bar(range(100), spike_counts, color='steelblue')
            ax.set_title(f'Digit {digit}\nHidden Layer Response')
            ax.set_xlabel('Neuron Index')
            ax.set_ylabel('Spike Count')
            ax.grid(True, alpha=0.3)
            
            # Mark top neurons
            top_idx = np.argsort(-spike_counts)[:5]
            for idx in top_idx:
                ax.axvline(idx, color='red', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('hidden_layer_digit_response.png', dpi=100)
    print("\nSaved: hidden_layer_digit_response.png")
    plt.close()


def example_2_hidden_neuron_specialization():
    """Example 2: Show which hidden neurons specialize for which digits"""
    print("\n" + "="*70)
    print("Example 2: Hidden Layer Specialization")
    print("="*70)
    
    print("\n[1/3] Loading and preparing data...")
    X_train, y_train, _, _ = load_mnist_28x28_small(num_train=200, num_test=50)
    X_train = preprocess_data(X_train.reshape(len(X_train), -1), normalization='minmax')
    X_train = X_train.reshape(-1, 28, 28)
    
    print("[2/3] Creating and training network...")
    network = STDPNetworkWithHiddenClassifier(
        input_width=28,
        input_height=28,
        hidden_size=100,
        num_classes=10
    )
    
    network.train(X_train, y_train, epochs=5, batch_size=20, duration=100)
    
    print("[3/3] Visualizing specialization...")
    
    # Get specialization
    specialization = network.get_hidden_layer_specialization()
    
    # Create heatmap of digit-neuron associations
    assoc_matrix = np.zeros((10, 100))
    for digit in range(10):
        neurons = specialization[digit]['neurons']
        weights = specialization[digit]['weights']
        for neuron, weight in zip(neurons, weights):
            assoc_matrix[digit, neuron] = weight
    
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(assoc_matrix, cmap='hot', aspect='auto')
    ax.set_xlabel('Hidden Neuron Index')
    ax.set_ylabel('Digit')
    ax.set_yticks(range(10))
    ax.set_title('Hidden Layer Specialization: Which Neurons Respond to Which Digits')
    plt.colorbar(im, ax=ax, label='Association Weight')
    plt.tight_layout()
    plt.savefig('hidden_layer_specialization.png', dpi=100)
    print("\nSaved: hidden_layer_specialization.png")
    plt.close()


def example_3_prediction_details():
    """Example 3: Show detailed prediction process"""
    print("\n" + "="*70)
    print("Example 3: Prediction Details")
    print("="*70)
    
    print("\n[1/4] Loading and preparing data...")
    X_train, y_train, X_test, y_test = load_mnist_28x28_small(num_train=250, num_test=50)
    X_train = preprocess_data(X_train.reshape(len(X_train), -1), normalization='minmax')
    X_test = preprocess_data(X_test.reshape(len(X_test), -1), normalization='minmax')
    X_train = X_train.reshape(-1, 28, 28)
    X_test = X_test.reshape(-1, 28, 28)
    
    print("[2/4] Creating and training network...")
    network = STDPNetworkWithHiddenClassifier(
        input_width=28,
        input_height=28,
        hidden_size=100,
        num_classes=10
    )
    
    network.train(X_train, y_train, epochs=5, batch_size=20, duration=100)
    
    print("[4/4] Analyzing predictions...")
    
    # Test on specific digits
    test_digits = [0, 3, 5, 8]
    fig, axes = plt.subplots(len(test_digits), 3, figsize=(15, 4*len(test_digits)))
    
    for row, target_digit in enumerate(test_digits):
        # Find a test sample for this digit
        digit_samples_idx = np.where(y_test == target_digit)[0]
        if len(digit_samples_idx) == 0:
            continue
        
        test_sample = X_test[digit_samples_idx[0]]
        true_label = y_test[digit_samples_idx[0]]
        
        # Get prediction
        spike_train = convert_to_poisson_spike_train(
            test_sample.flatten(), duration=100, dt=0.1
        )
        predicted_digit, details = network.predict(spike_train, duration=100)
        
        # Plot 1: Input image
        ax = axes[row, 0]
        ax.imshow(test_sample, cmap='gray')
        ax.set_title(f'Input: Digit {true_label}\nPredicted: {predicted_digit}')
        ax.axis('off')
        
        # Plot 2: Hidden layer activity
        ax = axes[row, 1]
        spike_counts = details['hidden_spike_counts']
        colors = ['red' if i in details['top_hidden_neurons'] else 'steelblue' 
                 for i in range(100)]
        ax.bar(range(100), spike_counts, color=colors)
        ax.set_title('Hidden Layer Spike Counts\n(red = top 10 neurons)')
        ax.set_xlabel('Neuron Index')
        ax.set_ylabel('Spike Count')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Class scores
        ax = axes[row, 2]
        class_scores = details['class_scores']
        colors_class = ['green' if i == predicted_digit else 'lightgray' for i in range(10)]
        ax.bar(range(10), class_scores, color=colors_class)
        ax.set_title('Class Scores')
        ax.set_xlabel('Digit Class')
        ax.set_ylabel('Score')
        ax.set_xticks(range(10))
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_details.png', dpi=100)
    print("\nSaved: prediction_details.png")
    plt.close()


def example_4_training_convergence():
    """Example 4: Show training convergence"""
    print("\n" + "="*70)
    print("Example 4: Training Convergence")
    print("="*70)
    
    print("\n[1/3] Loading and preparing data...")
    X_train, y_train, X_test, y_test = load_mnist_28x28_small(num_train=300, num_test=100)
    X_train = preprocess_data(X_train.reshape(len(X_train), -1), normalization='minmax')
    X_test = preprocess_data(X_test.reshape(len(X_test), -1), normalization='minmax')
    X_train = X_train.reshape(-1, 28, 28)
    X_test = X_test.reshape(-1, 28, 28)
    
    print("[2/3] Training network...")
    network = STDPNetworkWithHiddenClassifier(
        input_width=28,
        input_height=28,
        hidden_size=100,
        num_classes=10
    )
    
    network.train(X_train, y_train, epochs=10, batch_size=30, duration=100)
    
    print("[3/3] Evaluating and visualizing...")
    
    # Evaluate on test set
    test_accuracy, test_results = network.evaluate(X_test, y_test, duration=100)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training accuracy
    ax = axes[0]
    ax.plot(network.train_history['accuracy'], marker='o', linewidth=2, markersize=8, label='Training')
    ax.axhline(test_accuracy, color='red', linestyle='--', linewidth=2, label=f'Test Accuracy: {test_accuracy:.4f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Network Training Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Confusion matrix-like visualization
    ax = axes[1]
    predictions = np.array(test_results['predictions'])
    
    # Create a simple confusion matrix
    confusion = np.zeros((10, 10))
    for true_label, pred_label in zip(y_test, predictions):
        confusion[true_label, pred_label] += 1
    
    im = ax.imshow(confusion, cmap='Blues', aspect='auto')
    ax.set_xlabel('Predicted Digit')
    ax.set_ylabel('True Digit')
    ax.set_title('Prediction Distribution')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    plt.colorbar(im, ax=ax, label='Count')
    
    plt.tight_layout()
    plt.savefig('training_convergence.png', dpi=100)
    print("\nSaved: training_convergence.png")
    plt.close()


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("Hidden Layer Based Digit Recognition Examples")
    print("="*70)
    print("\nArchitecture:")
    print("  Input: 28×28 = 784 neurons (Poisson encoding)")
    print("  Hidden: 100 neurons (STDP learning)")
    print("  Classification: Based on hidden layer neuron activity")
    
    np.random.seed(42)
    
    try:
        example_1_single_digit_analysis()
        example_2_hidden_neuron_specialization()
        example_3_prediction_details()
        example_4_training_convergence()
        
        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("Generated visualizations:")
        print("  - hidden_layer_digit_response.png")
        print("  - hidden_layer_specialization.png")
        print("  - prediction_details.png")
        print("  - training_convergence.png")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
