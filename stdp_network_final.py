"""
Final STDP Network Implementation:
- Input: 28x28 neurons with Poisson spiking (784 neurons)
- Hidden: 100 neurons (fully connected from input, STDP learning)
  → Recognition is based on this layer's activity
- Output: 100 neurons (local 1-to-1 connectivity + lateral inhibition for processing)

Architecture:
Input (784) → Hidden (100) → Output (100 with lateral inhibition)
             ↑
         [Recognition based on hidden layer firing]
"""

import numpy as np
from neuron import PopulationNeurons
from synapse import SynapticLayer
from dataset import convert_to_poisson_spike_train


class LateralInhibitionLayer:
    """
    Third layer with lateral inhibition for signal processing
    
    Properties:
    - Each output neuron i receives input from hidden neuron i (1-to-1 mapping)
    - Each output neuron inhibits all other 99 output neurons
    - Used for feature refinement, NOT for digit recognition
    """
    
    def __init__(self, num_neurons=100, tau_m=10.0, tau_s=5.0, dt=0.1):
        self.num_neurons = num_neurons
        self.neurons = PopulationNeurons(num_neurons, tau_m, tau_s, threshold=0.5, dt=dt)
        self.dt = dt
        
        # 1-to-1 connectivity weights from hidden to output
        # Each output neuron i is mainly driven by hidden neuron i
        self.weights = np.zeros((num_neurons, num_neurons))
        np.fill_diagonal(self.weights, np.random.uniform(0.7, 0.9, num_neurons))  # Strong diagonal
        
        # Off-diagonal weak connections
        for i in range(num_neurons):
            for j in range(num_neurons):
                if i != j and np.random.rand() < 0.1:  # 10% probability
                    self.weights[i, j] = np.random.uniform(0.1, 0.3)
        
        # Lateral inhibition weights (off-diagonal only)
        self.inhibition_weights = np.ones((num_neurons, num_neurons)) - np.eye(num_neurons)
        self.inhibition_strength = 0.5
    
    def forward(self, hidden_spikes):
        """
        Forward pass with lateral inhibition
        
        Parameters:
        -----------
        hidden_spikes : array
            Spikes from hidden layer (100,)
            
        Returns:
        --------
        array
            Output spikes with lateral inhibition (100,)
        """
        # Compute postsynaptic currents from hidden layer
        psc = np.dot(self.weights, hidden_spikes)
        
        # Apply lateral inhibition iteratively
        current_spikes = np.zeros(self.num_neurons, dtype=int)
        
        for iteration in range(5):
            # Lateral inhibition from other neurons
            inhibition = np.dot(self.inhibition_weights, current_spikes) * self.inhibition_strength
            total_input = psc - inhibition
            
            # Integrate and check threshold
            self.neurons.integrate_inputs(total_input)
            spikes = self.neurons.check_thresholds(iteration * self.dt)
            current_spikes = current_spikes | spikes
        
        return current_spikes
    
    def reset(self):
        """Reset layer state"""
        self.neurons.reset()
    
    def get_spike_times(self):
        """Get spike times for analysis"""
        return self.neurons.get_spikes()


class STDPNetworkFinal:
    """
    Final STDP Network for digit recognition based on hidden layer activity
    
    Architecture:
    - Input: 28×28 = 784 neurons (Poisson encoding)
    - Hidden: 100 neurons (STDP learning) ← RECOGNITION BASED ON THIS LAYER
    - Output: 100 neurons (lateral inhibition for processing)
    
    Recognition method:
    - Count spikes in hidden layer for test digit
    - Map hidden neuron indices to digit classes (0-9)
    - Higher total spike count = higher confidence for that digit
    """
    
    def __init__(self, input_width=28, input_height=28, hidden_size=100, output_size=100,
                 tau_m=10.0, tau_s=5.0, tau_plus=20.0, tau_minus=20.0,
                 A_plus=0.01, A_minus=0.01, dt=0.1):
        """
        Parameters:
        -----------
        input_width, input_height : int
            Input image dimensions (28x28)
        hidden_size : int
            Number of hidden neurons (100)
        output_size : int
            Number of output neurons (100)
        """
        self.input_width = input_width
        self.input_height = input_height
        self.input_size = input_width * input_height  # 784
        self.hidden_size = hidden_size  # 100
        self.output_size = output_size  # 100
        self.num_classes = 10
        self.dt = dt
        
        print("\n" + "="*70)
        print("STDP Network Architecture:")
        print("="*70)
        print(f"Input Layer:    28×28 = {self.input_size} neurons")
        print(f"Hidden Layer:   {hidden_size} neurons (← digit recognition based on this)")
        print(f"Output Layer:   {output_size} neurons (lateral inhibition)")
        print("="*70 + "\n")
        
        # Neuron populations
        self.input_neurons = PopulationNeurons(self.input_size, tau_m, tau_s, dt=dt)
        self.hidden_neurons = PopulationNeurons(
            hidden_size, tau_m, tau_s, threshold=0.5, dt=dt
        )
        self.output_layer = LateralInhibitionLayer(output_size, tau_m, tau_s, dt)
        
        # Synaptic connections: Input → Hidden (STDP learning)
        self.input_to_hidden = SynapticLayer(
            self.input_size, hidden_size,
            tau_plus=tau_plus, tau_minus=tau_minus,
            A_plus=A_plus, A_minus=A_minus,
            connectivity='full'
        )
        
        # Synaptic connections: Hidden → Output (1-to-1 + weak connections)
        self.hidden_to_output = SynapticLayer(
            hidden_size, output_size,
            tau_plus=tau_plus, tau_minus=tau_minus,
            A_plus=A_plus, A_minus=A_minus,
            connectivity='full'
        )
        
        # Initialize hidden-to-output weights with 1-to-1 connectivity
        # (output neuron i receives most from hidden neuron i)
        weights = np.zeros((output_size, hidden_size))
        np.fill_diagonal(weights[:min(output_size, hidden_size), :hidden_size], 
                         np.random.uniform(0.7, 0.9, min(output_size, hidden_size)))
        self.hidden_to_output.set_weights(weights)
        
        # Hidden-to-class mapping for digit recognition
        # This maps hidden neuron activity to digit classes
        self.hidden_to_class = np.zeros((self.num_classes, hidden_size))
        
        # Training history
        self.train_history = {
            'accuracy': []
        }
    
    def forward(self, spike_train, duration=100):
        """
        Forward pass through network
        
        Parameters:
        -----------
        spike_train : array
            Input spike train (784, num_timesteps)
        duration : int
            Simulation duration in ms
            
        Returns:
        --------
        dict
            Network state including hidden and output spikes
        """
        num_steps = spike_train.shape[1]
        
        # Reset all neurons
        self.input_neurons.reset()
        self.hidden_neurons.reset()
        self.output_layer.reset()
        
        # Simulate network
        for t in range(num_steps):
            time_ms = t * self.dt
            
            # Layer 1: Input (directly from spike train)
            input_spikes = spike_train[:, t].astype(int)
            
            # Layer 2: Hidden (STDP learning happens here during training)
            hidden_input = self.input_to_hidden.forward(input_spikes)
            self.hidden_neurons.integrate_inputs(hidden_input)
            hidden_spikes = self.hidden_neurons.check_thresholds(time_ms)
            
            # Layer 3: Output (lateral inhibition processing)
            output_input = self.hidden_to_output.forward(hidden_spikes)
            # Note: output layer is not used for recognition, only for processing
            output_spikes = self.output_layer.forward(hidden_spikes)
            
            # Record output spikes for STDP learning if needed
            for i, spike in enumerate(output_spikes):
                if spike:
                    self.output_layer.neurons.spike_times[i].append(time_ms)
        
        return {
            'hidden_spikes': self.hidden_neurons.get_spikes(),
            'output_spikes': self.output_layer.get_spike_times(),
            'hidden_spike_counts': np.array([len(s) for s in self.hidden_neurons.get_spikes()])
        }
    
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
            output_input = self.hidden_to_output.forward(hidden_spikes)
            output_spikes = self.output_layer.forward(hidden_spikes)
            
            for i, spike in enumerate(output_spikes):
                if spike:
                    self.output_layer.neurons.spike_times[i].append(time_ms)
        
        # Get spike times
        hidden_spike_times = self.hidden_neurons.get_spikes()
        input_spike_times = self.input_neurons.get_spikes()
        
        # STDP learning: Update Input→Hidden weights
        self.input_to_hidden.update_stdp(input_spike_times, hidden_spike_times)
        
        # Update hidden-to-class mapping
        # This strengthens the association between active hidden neurons and the digit label
        hidden_spike_counts = np.array([len(spikes) for spikes in hidden_spike_times])
        
        # Reinforce hidden neurons that fired for this digit
        alpha = 0.01  # Learning rate
        self.hidden_to_class[label] += alpha * hidden_spike_counts
        self.hidden_to_class = np.clip(self.hidden_to_class, 0, 1)
        
        # Optional: STDP learning for Hidden→Output layer
        output_spike_times = self.output_layer.get_spike_times()
        self.hidden_to_output.update_stdp(hidden_spike_times, output_spike_times)
    
    def predict(self, spike_train, duration=100):
        """
        Predict digit based on HIDDEN LAYER activity
        
        Recognition method:
        1. Get spike counts from hidden layer neurons
        2. Calculate class scores using learned associations
        3. Return digit with highest score
        
        Parameters:
        -----------
        spike_train : array
            Input spike train (784, num_timesteps)
        duration : int
            Simulation duration in ms
            
        Returns:
        --------
        int : predicted digit (0-9)
        dict : detailed prediction information
        """
        # Get network response
        result = self.forward(spike_train, duration)
        hidden_spike_counts = result['hidden_spike_counts']
        
        # Calculate class scores
        class_scores = np.dot(self.hidden_to_class, hidden_spike_counts)
        
        # If no learned associations yet, use simple heuristic
        if class_scores.sum() == 0:
            # Group hidden neurons by position
            # Neurons 0-10 vote for digit 0, 11-20 for digit 1, etc.
            digit_scores = np.zeros(self.num_classes)
            for digit in range(self.num_classes):
                start = digit * (self.hidden_size // self.num_classes)
                end = (digit + 1) * (self.hidden_size // self.num_classes)
                digit_scores[digit] = hidden_spike_counts[start:end].sum()
            predicted_digit = np.argmax(digit_scores)
        else:
            predicted_digit = np.argmax(class_scores)
        
        return predicted_digit, {
            'hidden_spike_counts': hidden_spike_counts,
            'class_scores': class_scores,
            'top_neurons': np.argsort(-hidden_spike_counts)[:15].tolist(),
            'total_hidden_spikes': hidden_spike_counts.sum()
        }
    
    def train(self, X_train, y_train, epochs=10, batch_size=32, duration=100):
        """
        Train the network
        
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
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_accuracy = 0
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = batch_start + batch_size
                
                X_batch = X_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]
                
                for x, y in zip(X_batch, y_batch):
                    # Convert to spike train
                    x_flat = x.flatten()
                    spike_train = convert_to_poisson_spike_train(
                        x_flat, duration=duration, dt=self.dt
                    )
                    
                    # Learn with STDP
                    self.learn(spike_train, y, duration=duration)
                    
                    # Predict
                    prediction, _ = self.predict(spike_train, duration=duration)
                    
                    if prediction == y:
                        epoch_accuracy += 1
            
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
            Test data (num_samples, 28, 28)
        y_test : array
            Test labels (num_samples,)
        duration : int
            Simulation duration for each sample in ms
            
        Returns:
        --------
        float : accuracy
        dict : detailed results
        """
        correct = 0
        predictions = []
        details_list = []
        
        for x, y in zip(X_test, y_test):
            x_flat = x.flatten()
            spike_train = convert_to_poisson_spike_train(
                x_flat, duration=duration, dt=self.dt
            )
            
            prediction, details = self.predict(spike_train, duration=duration)
            predictions.append(prediction)
            details_list.append(details)
            
            if prediction == y:
                correct += 1
        
        accuracy = correct / len(X_test)
        
        return accuracy, {
            'predictions': predictions,
            'accuracy': accuracy,
            'details': details_list
        }
    
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
