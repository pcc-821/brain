"""
Comprehensive example script demonstrating STDP network usage
"""

import numpy as np
import matplotlib.pyplot as plt
from dataset import load_mnist_sklearn, preprocess_data, convert_to_poisson_spike_train
from stdp_network import STDPNetwork
from utils import (
    plot_spike_raster, plot_membrane_potential, plot_weight_matrix,
    plot_training_history, calculate_metrics, print_metrics
)


def example_basic_training():
    """Basic example: train and evaluate on subset"""
    print("="*60)
    print("Example 1: Basic Training and Evaluation")
    print("="*60)
    
    # Load data
    print("\n1. Loading dataset...")
    X_train, y_train, X_test, y_test = load_mnist_sklearn()
    
    # Use smaller subset for faster execution
    X_train = X_train[:300]
    y_train = y_train[:300]
    X_test = X_test[:100]
    y_test = y_test[:100]
    
    # Preprocess
    X_train = preprocess_data(X_train, normalization='minmax')
    X_test = preprocess_data(X_test, normalization='minmax')
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Create network
    print("\n2. Creating network...")
    network = STDPNetwork(
        input_size=X_train.shape[1],
        hidden_size=50,
        output_size=10,
        dt=1.0
    )
    print(f"   Input neurons: {network.input_size}")
    print(f"   Hidden neurons: {network.hidden_size}")
    print(f"   Output neurons: {network.output_size}")
    
    # Train
    print("\n3. Training network...")
    network.train(
        X_train, y_train,
        epochs=5,
        batch_size=10,
        duration=100
    )
    
    # Evaluate
    print("\n4. Evaluating...")
    accuracy = network.evaluate(X_test, y_test, duration=100)
    print(f"   Test Accuracy: {accuracy:.4f}")
    
    return network, X_train, y_train, X_test, y_test


def example_spike_visualization():
    """Example: visualize spike patterns"""
    print("\n" + "="*60)
    print("Example 2: Spike Pattern Visualization")
    print("="*60)
    
    # Load and preprocess one sample
    print("\n1. Loading sample...")
    X_train, _, _, _ = load_mnist_sklearn()
    sample = preprocess_data(X_train[0:1], normalization='minmax')[0]
    
    # Convert to spike train
    print("2. Converting to spike train...")
    spike_train = convert_to_poisson_spike_train(sample, duration=200, dt=1.0)
    print(f"   Spike train shape: {spike_train.shape}")
    print(f"   Firing rate: {spike_train.sum() / spike_train.size:.2%}")
    
    # Create network and run forward pass
    print("3. Running inference...")
    network = STDPNetwork(input_size=64, hidden_size=50, output_size=10)
    output_spikes, hidden_spikes, input_spikes = network.forward(spike_train, duration=200)
    
    # Visualize
    print("4. Creating visualizations...")
    
    # Plot input spike raster
    fig, ax = plot_spike_raster(input_spikes, title="Input Layer Spike Raster")
    plt.tight_layout()
    plt.savefig('spike_raster_input.png', dpi=100)
    print("   Saved: spike_raster_input.png")
    
    # Plot hidden spike raster
    fig, ax = plot_spike_raster(hidden_spikes, title="Hidden Layer Spike Raster")
    plt.tight_layout()
    plt.savefig('spike_raster_hidden.png', dpi=100)
    print("   Saved: spike_raster_hidden.png")
    
    plt.close('all')


def example_weight_analysis():
    """Example: analyze learned weights"""
    print("\n" + "="*60)
    print("Example 3: Weight Matrix Analysis")
    print("="*60)
    
    # Create and initialize network
    print("\n1. Creating network...")
    network = STDPNetwork(input_size=64, hidden_size=50, output_size=10)
    
    # Get initial weights
    print("2. Getting weight matrices...")
    weights = network.get_weights()
    
    # Visualize
    print("3. Creating visualizations...")
    
    fig, ax = plot_weight_matrix(
        weights['input_to_hidden'],
        title="Input → Hidden Weights (Initial)"
    )
    plt.tight_layout()
    plt.savefig('weights_input_to_hidden.png', dpi=100)
    print("   Saved: weights_input_to_hidden.png")
    
    fig, ax = plot_weight_matrix(
        weights['hidden_to_output'],
        title="Hidden → Output Weights (Initial)"
    )
    plt.tight_layout()
    plt.savefig('weights_hidden_to_output.png', dpi=100)
    print("   Saved: weights_hidden_to_output.png")
    
    plt.close('all')


def example_parameter_sensitivity():
    """Example: test sensitivity to parameters"""
    print("\n" + "="*60)
    print("Example 4: Parameter Sensitivity Analysis")
    print("="*60)
    
    # Load small dataset
    print("\n1. Loading dataset...")
    X_train, y_train, X_test, y_test = load_mnist_sklearn()
    X_train = X_train[:100]
    y_train = y_train[:100]
    X_test = X_test[:50]
    y_test = y_test[:50]
    
    X_train = preprocess_data(X_train, normalization='minmax')
    X_test = preprocess_data(X_test, normalization='minmax')
    
    # Test different hidden sizes
    print("\n2. Testing different hidden layer sizes...")
    hidden_sizes = [30, 50, 100]
    results = []
    
    for hidden_size in hidden_sizes:
        print(f"   Hidden size: {hidden_size}")
        
        network = STDPNetwork(
            input_size=64,
            hidden_size=hidden_size,
            output_size=10
        )
        
        network.train(X_train, y_train, epochs=3, batch_size=10, duration=100)
        accuracy = network.evaluate(X_test, y_test, duration=100)
        results.append(accuracy)
        
        print(f"      Accuracy: {accuracy:.4f}")
    
    # Plot results
    print("\n3. Visualizing results...")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hidden_sizes, results, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Hidden Layer Size')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Parameter Sensitivity Analysis')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('parameter_sensitivity.png', dpi=100)
    print("   Saved: parameter_sensitivity.png")
    
    plt.close('all')


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("STDP Network Examples")
    print("="*60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run examples
    try:
        # Example 1: Basic training
        network, X_train, y_train, X_test, y_test = example_basic_training()
        
        # Example 2: Spike visualization
        example_spike_visualization()
        
        # Example 3: Weight analysis
        example_weight_analysis()
        
        # Example 4: Parameter sensitivity
        example_parameter_sensitivity()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("Check the generated PNG files for visualizations.")
        print("="*60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
