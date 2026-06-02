"""
Specialized STDP Network with custom architecture:
- Input: 28x28 neurons with Poisson spiking
- Hidden: 100 neurons (fully connected from input)
- Output: 100 neurons (local connectivity + lateral inhibition)
"""

import numpy as np
from neuron import PopulationNeurons, LIFNeuron
from synapse import SynapticLayer
from dataset import convert_to_poisson_spike_train


class LateralInhibitionLayer:
    """
    Output layer with specialized connectivity:
    - Each neuron receives from one input neuron (1-to-1 mapping)
    - Each neuron inhibits all other 99 neurons (lateral inhibition)
    """
    
    def __init__(self, num_neurons=100, tau_m=10.0, tau_s=5.0, 
                 tau_plus=20.0, tau_minus=20.0, A_plus=0.01, A_minus=0.01, dt=0.1):
        """
        Parameters:
        -----------
        num_neurons : int
            Number of neurons in this layer (100)
        """
        self.num_neurons = num_neurons
        self.neurons = PopulationNeurons(num_neurons, tau_m, tau_s, threshold=0.5, dt=dt)
        
        # STDP parameters for learning
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus
        
        # Weights from 100 hidden neurons to 100 output neurons
        # Shape: (100, 100) - dense connectivity
        self.weights = np.random.uniform(0.1, 0.9, (num_neurons, num_neurons))
        
        # Lateral inhibition weights (100, 100)
        # Diagonal is 0 (no self-inhibition), off-diagonal is inhibitory
        self.inhibition_weights = np.ones((num_neurons, num_neurons)) - np.eye(num_neurons)
        self.inhibition_strength = 0.5  # Strength of lateral inhibition
        
    def forward(self, hidden_spikes, hidden_spike_times=None):
        """
        Forward pass with STDP learning
        
        Parameters:
        -----------
        hidden_spikes : array
            Spikes from hidden layer (100,)
        hidden_spike_times : list
            Spike times for STDP learning
            
        Returns:
        --------
        array
            Output spikes (100,)
        """
        # Compute postsynaptic currents from hidden layer
        psc = np.dot(self.weights, hidden_spikes)
        
        # Get current output spikes
        current_spikes = np.zeros(self.num_neurons, dtype=int)
        
        # Apply lateral inhibition iteratively
        # Each neuron that fires inhibits others
        for iteration in range(5):  # Multiple iterations for inhibition to take effect
            # Integrate inputs with inhibition
            inhibition = np.dot(self.inhibition_weights, current_spikes) * self.inhibition_strength
            total_input = psc - inhibition
            
            # Apply to neurons
            self.neurons.integrate_inputs(total_input)
            
            # Check thresholds
            time_ms = iteration * 0.1
            spikes = self.neurons.check_thresholds(time_ms)
            current_spikes = current_spikes | spikes  # Accumulate spikes
        
        return current_spikes
    
    def learn_stdp(self, hidden_spike_times, output_spike_times):
        """
        Apply STDP learning
        
        Parameters:
        -----------
        hidden_spike_times : list of lists
            Spike times for each hidden neuron
        output_spike_times : list of lists
            Spike times for each output neuron
        """
        for out_idx, out_spikes in enumerate(output_spike_times):
            for hid_idx, hid_spikes in enumerate(hidden_spike_times):
                if len(out_spikes) == 0 or len(hid_spikes) == 0:
                    continue
                
                # STDP learning
                for t_out in out_spikes:
                    for t_hid in hid_spikes:
                        delta_t = t_out - t_hid
                        
                        if delta_t > 0:  # LTP
                            weight_change = self.A_plus * np.exp(-delta_t / self.tau_plus)
                        else:  # LTD
                            weight_change = -self.A_minus * np.exp(delta_t / self.tau_minus)
                        
                        self.weights[out_idx, hid_idx] += weight_change
        
        # Clip weights
        self.weights = np.clip(self.weights, 0.0, 1.0)
    
    def get_weights(self):
        """Get weight matrix"""
        return self.weights
    
    def set_weights(self, weights):
        """Set weight matrix"""
        self.weights = np.clip(weights, 0.0, 1.0)
    
    def reset(self):
        """Reset layer state"""
        self.neurons.reset()


class STDPNetworkCustom:
    """
    Custom STDP Network:
    - Input layer: 28x28 = 784 neurons (Poisson encoding)
    - Hidden layer: 100 neurons (fully connected from input)
    - Output layer: 100 neurons (with lateral inhibition)
    """
    
    def __init__(self, input_width=28, input_height=28, hidden_size=100, output_size=100,
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
        output_size : int
            Number of output neurons (100)
        """
        self.input_width = input_width
        self.input_height = input_height
        self.input_size = input_width * input_height  # 784
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dt = dt
        
        # Neuron populations
        self.input_neurons = PopulationNeurons(self.input_size, tau_m, tau_s, dt=dt)
        self.hidden_neurons = PopulationNeurons(hidden_size, tau_m, tau_s, threshold=0.5, dt=dt)
        
        # Output layer with lateral inhibition
        self.output_layer = LateralInhibitionLayer(
            output_size, tau_m, tau_s, tau_plus, tau_minus, A_plus, A_minus, dt
        )
        
        # Synaptic layer from input to hidden (fully connected)
        self.input_to_hidden = SynapticLayer(
            self.input_size, hidden_size,
            tau_plus=tau_plus, tau_minus=tau_minus,
            A_plus=A_plus, A_minus=A_minus,
            connectivity='full'
        )
        
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
            (output_spikes, hidden_spikes, input_spikes)
        """
        num_steps = spike_train.shape[1]
        
        # Reset neurons
        self.input_neurons.reset()
        self.hidden_neurons.reset()
        self.output_layer.reset()
        
        # Simulate network
        for t in range(num_steps):
            time_ms = t * self.dt
            
            # Input layer (directly uses spike train)
            input_spikes = spike_train[:, t].astype(int)
            
            # Hidden layer
            hidden_input = self.input_to_hidden.forward(input_spikes)
            self.hidden_neurons.integrate_inputs(hidden_input)
            hidden_spikes = self.hidden_neurons.check_thresholds(time_ms)
            
            # Output layer with lateral inhibition
            output_spikes = self.output_layer.forward(hidden_spikes)
            # Record output spikes
            for i, spike in enumerate(output_spikes):
                if spike:
                    self.output_layer.neurons.spike_times[i].append(time_ms)
        
        return (
            self.output_layer.neurons.get_spikes(),
            self.hidden_neurons.get_spikes(),
            self.input_neurons.get_spikes()
        )
    
    def learn(self, spike_train, duration=100):
        """
        Forward pass with STDP learning
        
        Parameters:
        -----------
        spike_train : array
            Input spike train (784, num_timesteps)
        duration : int
            Simulation duration in ms
        """
        num_steps = spike_train.shape[1]
        
        # Reset neurons
        self.input_neurons.reset()
        self.hidden_neurons.reset()
        self.output_layer.reset()
        
        # Simulate network
        for t in range(num_steps):
            time_ms = t * self.dt
            
            # Input layer
            input_spikes = spike_train[:, t].astype(int)
            
            # Hidden layer
            hidden_input = self.input_to_hidden.forward(input_spikes)
            self.hidden_neurons.integrate_inputs(hidden_input)
            hidden_spikes = self.hidden_neurons.check_thresholds(time_ms)
            
            # Output layer
            output_spikes = self.output_layer.forward(hidden_spikes)
            for i, spike in enumerate(output_spikes):
                if spike:
                    self.output_layer.neurons.spike_times[i].append(time_ms)
        
        # Apply STDP learning
        self.input_to_hidden.update_stdp(
            self.input_neurons.get_spikes(),
            self.hidden_neurons.get_spikes()
        )
        
        self.output_layer.learn_stdp(
            self.hidden_neurons.get_spikes(),
            self.output_layer.neurons.get_spikes()
        )
    
    def predict(self, spike_train, duration=100):
        """
        Get network prediction (digit 0-9 based on output neuron activity)
        
        Parameters:
        -----------
        spike_train : array
            Input spike train (784, num_timesteps)
        duration : int
            Simulation duration in ms
            
        Returns:
        --------
        int
            Predicted digit (0-9) based on output neuron groups
        """
        output_spikes, _, _ = self.forward(spike_train, duration)
        
        # Divide 100 output neurons into 10 groups (10 neurons per digit)
        spike_counts = np.array([len(spikes) for spikes in output_spikes])
        
        # Sum spikes for each digit group
        digit_scores = np.zeros(10)
        for digit in range(10):
            start_idx = digit * 10
            end_idx = start_idx + 10
            digit_scores[digit] = spike_counts[start_idx:end_idx].sum()
        
        return np.argmax(digit_scores)
    
    def train(self, X_train, y_train, epochs=10, batch_size=32, duration=100):
        """
        Train the network
        
        Parameters:
        -----------
        X_train : array
            Training data (num_samples, 784)
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
                    spike_train = convert_to_poisson_spike_train(x_flat, duration=duration, dt=self.dt)
                    
                    # Learn
                    self.learn(spike_train, duration=duration)
                    
                    # Predict
                    prediction = self.predict(spike_train, duration=duration)
                    
                    # Track accuracy
                    if prediction == y:
                        epoch_accuracy += 1
            
            # Calculate metrics
            accuracy = epoch_accuracy / num_samples
            self.train_history['accuracy'].append(accuracy)
            
            if (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.4f}")
    
    def evaluate(self, X_test, y_test, duration=100):
        """
        Evaluate network on test data
        
        Parameters:
        -----------
        X_test : array
            Test data (num_samples, 784)
        y_test : array
            Test labels (num_samples,)
        duration : int
            Simulation duration for each sample in ms
            
        Returns:
        --------
        float
            Accuracy on test data
        """
        correct = 0
        
        for x, y in zip(X_test, y_test):
            # Convert to spike train
            x_flat = x.flatten()
            spike_train = convert_to_poisson_spike_train(x_flat, duration=duration, dt=self.dt)
            
            # Predict
            prediction = self.predict(spike_train, duration=duration)
            
            if prediction == y:
                correct += 1
        
        accuracy = correct / len(X_test)
        return accuracy
    
    def get_weights(self):
        """Get network weights"""
        return {
            'input_to_hidden': self.input_to_hidden.get_weights(),
            'hidden_to_output': self.output_layer.get_weights()
        }
    
    def set_weights(self, weights):
        """Set network weights"""
        self.input_to_hidden.set_weights(weights['input_to_hidden'])
        self.output_layer.set_weights(weights['hidden_to_output'])
