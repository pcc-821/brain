"""
STDP Network with digit recognition based on hidden layer neuron activity
- Input: 28x28 neurons with Poisson spiking
- Hidden: 100 neurons (fully connected from input, with STDP learning)
- Recognition: Based on which hidden neuron fires most for each digit
"""

import numpy as np
from neuron import PopulationNeurons
from synapse import SynapticLayer
from dataset import convert_to_poisson_spike_train


class STDPNetworkWithHiddenClassifier:
    """
    STDP Network with hidden layer based digit recognition
    
    The hidden layer neurons learn to specialize on different digits through STDP.
    Each hidden neuron develops sensitivity to specific digit patterns.
    During recognition, we count spikes in hidden layer and classify based on
    which hidden neurons fire most.
    """
    
    def __init__(self, input_width=28, input_height=28, hidden_size=100, num_classes=10,
                 tau_m=10.0, tau_s=5.0, tau_plus=20.0, tau_minus=20.0,
                 A_plus=0.01, A_minus=0.01, dt=0.1):
        """
        Parameters:
        -----------
        input_width : int
            Width of input image (28)
        input_height : int
            Height of input image (28)
        hidden_size : int
            Number of hidden neurons (100)
        num_classes : int
            Number of digit classes (10)
        """
        self.input_width = input_width
        self.input_height = input_height
        self.input_size = input_width * input_height  # 784
        self.hidden_size = hidden_size  # 100
        self.num_classes = num_classes  # 10 digits (0-9)
        self.dt = dt
        
        # Neuron populations
        self.input_neurons = PopulationNeurons(self.input_size, tau_m, tau_s, dt=dt)
        self.hidden_neurons = PopulationNeurons(
            hidden_size, tau_m, tau_s, threshold=0.5, dt=dt
        )
        
        # Synaptic layer from input to hidden (fully connected with STDP)
        self.input_to_hidden = SynapticLayer(
            self.input_size, hidden_size,
            tau_plus=tau_plus, tau_minus=tau_minus,
            A_plus=A_plus, A_minus=A_minus,
            connectivity='full'
        )
        
        # Classification layer: maps hidden neurons to digit classes
        # Shape: (num_classes, hidden_size) = (10, 100)
        # Each row represents which hidden neurons respond to each digit
        self.hidden_to_class = np.zeros((num_classes, hidden_size))
        
        # Training history
        self.train_history = {
            'loss': [],
            'accuracy': []
        }
    
    def forward(self, spike_train, duration=100):
        """
        Forward pass through the network
        
        Parameters:
        -----------
        spike_train : array
            Input spike train (784, num_timesteps)
        duration : int
            Simulation duration in ms
            
        Returns:
        --------
        tuple
            (hidden_spikes, spike_times for hidden layer)
        """
        num_steps = spike_train.shape[1]
        
        # Reset neurons
        self.input_neurons.reset()
        self.hidden_neurons.reset()
        
        # Simulate network
        for t in range(num_steps):
            time_ms = t * self.dt
            
            # Input layer (directly uses spike train)
            input_spikes = spike_train[:, t].astype(int)
            
            # Hidden layer
            hidden_input = self.input_to_hidden.forward(input_spikes)
            self.hidden_neurons.integrate_inputs(hidden_input)
            hidden_spikes = self.hidden_neurons.check_thresholds(time_ms)
        
        return self.hidden_neurons.get_spikes()
    
    def learn(self, spike_train, label, duration=100):
        """
        Forward pass with STDP learning
        
        Parameters:
        -----------
        spike_train : array
            Input spike train (784, num_timesteps)
        label : int
            Digit label (0-9)
        duration : int
            Simulation duration in ms
        """
        num_steps = spike_train.shape[1]
        
        # Reset neurons
        self.input_neurons.reset()
        self.hidden_neurons.reset()
        
        # Simulate network
        for t in range(num_steps):
            time_ms = t * self.dt
            
            # Input layer
            input_spikes = spike_train[:, t].astype(int)
            
            # Hidden layer
            hidden_input = self.input_to_hidden.forward(input_spikes)
            self.hidden_neurons.integrate_inputs(hidden_input)
            hidden_spikes = self.hidden_neurons.check_thresholds(time_ms)
        
        # Get spike times
        hidden_spike_times = self.hidden_neurons.get_spikes()
        input_spike_times = self.input_neurons.get_spikes()
        
        # Apply STDP learning at input-to-hidden synapses
        self.input_to_hidden.update_stdp(input_spike_times, hidden_spike_times)
        
        # Update classification weights: count spikes in hidden layer for this label
        hidden_spike_counts = np.array([len(spikes) for spikes in hidden_spike_times])
        
        # Reinforce hidden neurons that fired for this digit
        # Use exponential moving average to update weights
        alpha = 0.01  # Learning rate for classification layer
        self.hidden_to_class[label] += alpha * hidden_spike_counts
        
        # Normalize to keep weights bounded
        self.hidden_to_class = np.clip(self.hidden_to_class, 0, 1)
    
    def predict(self, spike_train, duration=100):
        """
        Predict digit based on hidden layer activity
        
        Method: Find which hidden neurons fire most, then find which digit
        class those neurons are most associated with
        
        Parameters:
        -----------
        spike_train : array
            Input spike train (784, num_timesteps)
        duration : int
            Simulation duration in ms
            
        Returns:
        --------
        int
            Predicted digit (0-9)
        dict
            Detailed prediction information
        """
        # Get hidden layer spikes
        hidden_spike_times = self.forward(spike_train, duration)
        hidden_spike_counts = np.array([len(spikes) for spikes in hidden_spike_times])
        
        # Method 1: Dot product between hidden activity and learned class weights
        class_scores = np.dot(self.hidden_to_class, hidden_spike_counts)
        
        # If no learned weights yet, use simple voting
        if class_scores.sum() == 0:
            # Find most active hidden neurons and assign to digit by position
            most_active_idx = np.argsort(-hidden_spike_counts)[:10]  # Top 10
            predicted_digit = np.median(np.minimum(most_active_idx, 9)).astype(int)
        else:
            predicted_digit = np.argmax(class_scores)
        
        # Return detailed info
        return predicted_digit, {
            'hidden_spike_counts': hidden_spike_counts,
            'class_scores': class_scores,
            'top_hidden_neurons': np.argsort(-hidden_spike_counts)[:10].tolist(),
            'top_hidden_spikes': np.sort(-hidden_spike_counts)[:10].tolist()
        }
    
    def train(self, X_train, y_train, epochs=10, batch_size=32, duration=100):
        """
        Train the network using STDP learning
        
        Parameters:
        -----------
        X_train : array
            Training data (num_samples, 28, 28)
        y_train : array
            Training labels (num_samples,)
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        duration : int
            Simulation duration for each sample in ms
        """
        num_samples = len(X_train)
        num_batches = num_samples // batch_size
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_accuracy = 0
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = batch_start + batch_size
                
                X_batch = X_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]
                
                # Process batch
                for i, (x, y) in enumerate(zip(X_batch, y_batch)):
                    # Convert to spike train (flatten 28x28 to 784)
                    x_flat = x.flatten()
                    spike_train = convert_to_poisson_spike_train(
                        x_flat, duration=duration, dt=self.dt
                    )
                    
                    # Learn with STDP
                    self.learn(spike_train, y, duration=duration)
                    
                    # Predict
                    prediction, _ = self.predict(spike_train, duration=duration)
                    
                    # Track accuracy
                    if prediction == y:
                        epoch_accuracy += 1
            
            # Calculate metrics
            accuracy = epoch_accuracy / num_samples
            self.train_history['accuracy'].append(accuracy)
            
            if (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.4f}")
    
    def evaluate(self, X_test, y_test, duration=100, verbose=False):
        """
        Evaluate network on test data
        
        Parameters:
        -----------
        X_test : array
            Test data (num_samples, 28, 28)
        y_test : array
            Test labels (num_samples,)
        duration : int
            Simulation duration for each sample in ms
        verbose : bool
            Print detailed results
            
        Returns:
        --------
        float
            Accuracy on test data
        dict
            Detailed results
        """
        correct = 0
        predictions = []
        details_list = []
        
        for x, y in zip(X_test, y_test):
            # Convert to spike train
            x_flat = x.flatten()
            spike_train = convert_to_poisson_spike_train(
                x_flat, duration=duration, dt=self.dt
            )
            
            # Predict
            prediction, details = self.predict(spike_train, duration=duration)
            predictions.append(prediction)
            details_list.append(details)
            
            if prediction == y:
                correct += 1
        
        accuracy = correct / len(X_test)
        
        if verbose:
            print(f"\nDetailed Evaluation Results:")
            print(f"Correct predictions: {correct}/{len(X_test)}")
            print(f"Accuracy: {accuracy:.4f}")
        
        return accuracy, {
            'predictions': predictions,
            'accuracy': accuracy,
            'details': details_list
        }
    
    def get_hidden_layer_specialization(self):
        """
        Analyze which hidden neurons specialize for which digits
        
        Returns:
        --------
        dict
            Mapping from digit to most responsive hidden neurons
        """
        specialization = {}
        
        for digit in range(self.num_classes):
            # Get weights for this digit class
            weights = self.hidden_to_class[digit]
            
            # Find top responsive hidden neurons
            top_indices = np.argsort(-weights)[:10]  # Top 10
            top_weights = weights[top_indices]
            
            specialization[digit] = {
                'neurons': top_indices.tolist(),
                'weights': top_weights.tolist(),
                'weight_sum': weights.sum()
            }
        
        return specialization
    
    def get_weights(self):
        """Get network weights"""
        return {
            'input_to_hidden': self.input_to_hidden.get_weights(),
            'hidden_to_class': self.hidden_to_class.copy()
        }
    
    def set_weights(self, weights):
        """Set network weights"""
        self.input_to_hidden.set_weights(weights['input_to_hidden'])
        self.hidden_to_class = weights['hidden_to_class'].copy()
