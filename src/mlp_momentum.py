import numpy as np
import matplotlib.pyplot as plt
import time

class MLP:
    def __init__(self, train_data, target, num_epochs=100, num_input=2, num_hidden=2, num_output=1, lr=0.1, momentum=0.9):
        self.train_data = train_data
        self.target = target
        self.num_epochs = num_epochs
        self.lr = lr
        self.momentum = momentum
        
        # Initialize weights randomly
        self.input_hidden_weights = np.random.uniform(size=(num_input, num_hidden))
        self.hidden_output_weights = np.random.uniform(size=(num_hidden, num_output))

        # Initialize biases
        self.hidden_bias = np.random.uniform(size=(1,num_hidden))
        self.output_bias = np.random.uniform(size=(1,num_output))

        # Initialize plot lists
        self.mse = []
        self.hidden_mse = []
        self.classification_errors = []
        self.input_hidden_weights_history = []
        self.hidden_output_weights_history = []

    def update_weights(self):
        # Calculate MSE on the hidden layer
        hidden_prediction_error = (self.target - self.output_final) @ self.hidden_output_weights.T
        hidden_mse = np.mean(np.square(hidden_prediction_error))
        self.hidden_mse.append(hidden_mse)
        
        prediction_error = self.target - self.output_final

        # Calculate the gradients of the weights connecting the input layer to the hidden layer
        input_hidden_weight_gradients = self.train_data.T @ (((prediction_error * self.sigmoid_der(self.output_final)) * self.hidden_output_weights.T) * self.sigmoid_der(self.hidden_output))
        # Calculate the gradients of the weights connecting the hidden layer to the output layer
        hidden_output_weight_gradients = self.hidden_output.T @ (prediction_error * self.sigmoid_der(self.output_final))

        # Update the weights with momentum
        self.input_hidden_weights_delta = self.lr * input_hidden_weight_gradients + self.momentum * getattr(self, 'input_hidden_weights_delta', 0)
        self.input_hidden_weights += self.input_hidden_weights_delta

        self.hidden_output_weights_delta = self.lr * hidden_output_weight_gradients + self.momentum * getattr(self, 'hidden_output_weights_delta', 0)
        self.hidden_output_weights += self.hidden_output_weights_delta

        # Update the biases of the neurons in the hidden layer
        self.hidden_bias += np.sum(self.lr * ((prediction_error * self.sigmoid_der(self.output_final)) * self.hidden_output_weights.T) * self.sigmoid_der(self.hidden_output), axis=0)
        # Update the biases of the neurons in the output layer
        self.output_bias += np.sum(self.lr * prediction_error * self.sigmoid_der(self.output_final), axis=0)

        # Append current weights to history
        self.input_hidden_weights_history.append(self.input_hidden_weights.copy())
        self.hidden_output_weights_history.append(self.hidden_output_weights.copy())

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_der(self, x):
        return x * (1 - x)

    def forward(self, input_data):
        # Calculate the input to the hidden layer
        self.hidden_input = input_data @ self.input_hidden_weights + self.hidden_bias
        # Apply the sigmoid activation function to the hidden layer input
        self.hidden_output = self.sigmoid(self.hidden_input)

        # Calculate the input to the output layer
        self.output_input = self.hidden_output @ self.hidden_output_weights + self.output_bias
        # Apply the sigmoid activation function to the output layer input
        self.output_final = self.sigmoid(self.output_input)

        return self.output_final

    def classify(self, datapoint):
        datapoint = np.transpose(datapoint)
        output = self.forward(datapoint)
        return 1 if output >= 0.5 else 0
    
    def train(self, min_mse=0):
        for _ in range(self.num_epochs):
            self.forward(self.train_data)
             # Calculate MSE
            current_mse = np.mean(np.square(self.target - self.output_final))
            self.mse.append(current_mse)
            self.update_weights()
            errors = 0
            for i in range(len(self.train_data)):
                if self.classify(self.train_data[i]) != self.target[i]:
                    errors += 1
            self.classification_errors.append(errors)

            if current_mse <= min_mse:
                break
                
    def test(self, X_test):
        y_pred = self.forward(X_test)
        print("Input    |   Output")
        for i in range(len(X_test)):
            print(f"{X_test[i].tolist()}   |   {y_pred[i].item()}")

    def plot_mse(self):
        plt.plot(range(len(self.mse)), self.mse)
        plt.title('Mean Squared Error (MSE) Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.show()

    def plot_all_mse(self):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(len(self.mse)), self.mse)
        plt.title('Mean Squared Error (MSE) Over Epochs - Output')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.grid()

        plt.subplot(1, 2, 2)
        
        plt.plot(range(len(self.hidden_mse)), self.hidden_mse)
        plt.title('Mean Squared Error (MSE) Over Epochs - Hidden Layer')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.grid()

        plt.tight_layout()
        plt.show()
        
    def plot_classification_errors(self):
        plt.plot(range(len(self.classification_errors)), self.classification_errors)
        plt.title('Classification Errors Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Misclassified Points')
        plt.grid()
        plt.show()

    def plot_all(self):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(len(self.mse)), self.mse)
        plt.title('Mean Squared Error (MSE) Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(range(len(self.classification_errors)), self.classification_errors)
        plt.title('Classification Errors Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Misclassified Points')
        plt.grid()
        plt.show()

    def plot_decision_boundary(self):
        # Define the range for the meshgrid
        x_min, x_max = -0.5, 1.5
        y_min, y_max = -0.5, 1.5
        h = 0.01  # step size in the mesh

        # Create a meshgrid of points
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Predict the class for each point in the meshgrid
        Z = np.array([self.classify(np.array([xx.ravel()[i], yy.ravel()[i]])) for i in range(len(xx.ravel()))])
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary
        plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.5)

        # Plot the training points
        colors = ['red' if target == 0 else 'green' for target in self.target.flatten()]
        plt.scatter(self.train_data[:, 0], self.train_data[:, 1], c=colors, edgecolors='k', s=150)
        plt.xlabel('Input 1')
        plt.ylabel('Input 2')
        plt.title('Decision Boundary')
        plt.show()

    def plot_weights(self):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(self.input_hidden_weights, cmap='viridis', aspect='auto')
        plt.title('Input-Hidden Weights')
        plt.xlabel('Hidden Neurons')
        plt.ylabel('Input Neurons')
        plt.grid()
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(self.hidden_output_weights, cmap='viridis', aspect='auto')
        plt.title('Hidden-Output Weights')
        plt.xlabel('Output Neurons')
        plt.ylabel('Hidden Neurons')
        plt.grid()
        plt.colorbar()

        plt.tight_layout()
        plt.show()
    
    def plot_weights(self):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        for i in range(len(self.input_hidden_weights_history[0])):
            for j in range(len(self.input_hidden_weights_history[0][0])):
                weights = [epoch[i][j] for epoch in self.input_hidden_weights_history]
                plt.plot(range(len(weights)), weights, label=f'Input {i+1} to Hidden {j+1}')
        plt.title('Input-Hidden Weights History')
        plt.xlabel('Epochs')
        plt.ylabel('Weights')
        plt.grid()
        plt.legend()

        plt.subplot(1, 2, 2)
        for i in range(len(self.hidden_output_weights_history[0])):
            for j in range(len(self.hidden_output_weights_history[0][0])):
                weights = [epoch[i][j] for epoch in self.hidden_output_weights_history]
                plt.plot(range(len(weights)), weights, label=f'Hidden {i+1} to Output {j+1}')
        plt.title('Hidden-Output Weights History')
        plt.xlabel('Epochs')
        plt.ylabel('Weights')
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()

''' Main function '''
def main():

    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    start_time = time.time()
    mlp = MLP(train_data=X, target=y, num_epochs=2000, num_input=2, num_hidden=2, num_output=1, lr=0.1, momentum=0.9)
    mlp.train()
    end_time = time.time()
    print(f'Elapsed time: {end_time - start_time}s')
    print(f"Last MSE value: {mlp.mse[-1]}")

    # XOR test
    X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    mlp.test(X_test)

    # Plotting
    #mlp.plot_mse()
    #mlp.plot_all_mse()
    #mlp.plot_classification_errors()
    #mlp.plot_all()
    #mlp.plot_decision_boundary()
    mlp.plot_weights()

if __name__ == "__main__":
    main()
