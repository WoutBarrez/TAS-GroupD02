## -----------------------------------------------------------------------------------------------
## This file contains the code to train the classifier. This template was obtained from the
## AE2224-II Artificial Intelligence for Aerospace Engineering course taught in the second year
## of the Aerospace Engineering bachelor at TU Delft.
## -----------------------------------------------------------------------------------------------
import numpy as np
import torch

# Set the PyTorch and numpy random seeds for reproducibility:
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)


class MLPClassifier(torch.nn.Module):

    def __init__(self, n_features, n_hidden_neurons, n_classes, learning_rate, n_epochs):
        """
        Initialize the neural network classifier 
        """
        # initialize superclass
        super(MLPClassifier,self).__init__()

        # the number of classes
        self.n_classes = n_classes

        # the number of epochs
        self.n_epochs = n_epochs
        
        # create the neural network
        self.network = torch.nn.Sequential(torch.nn.Linear(n_features, n_hidden_neurons),
                            torch.nn.ReLU(),torch.nn.Linear(n_hidden_neurons,n_hidden_neurons),torch.nn.ReLU(),torch.nn.Linear(n_hidden_neurons,n_classes))



        # the cross-entropy loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # the Adam optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)


    def forward(self, X):
        """
        Forward pass of the neural network
        """
        return self.network(X)


    def train(self, X_train, y_train):
        """
        Train the neural network classifier 
        """

        # convert data into appropriate format for {torch}
        X_train_torch = torch.tensor(X_train, dtype=torch.float32)
        y_train_torch = torch.nn.functional.one_hot(torch.tensor(y_train), num_classes=self.n_classes).to(dtype=torch.float32)

        # start training
        print("Training started...")
        for epoch in range(0, self.n_epochs):
            # Set optimizer to zero grad
            self.optimizer.zero_grad()

            # Forward step
            y_pred = self.forward(X_train_torch)

            # Compute loss
            loss = self.loss_fn(y_pred, y_train_torch)

            # Backward step
            loss.backward()

            # Update weights
            self.optimizer.step()

            # Report loss
            if not ((epoch+1) % 100):
                print("Epoch:", epoch+1, f", loss = {loss.detach().item():.4f} ")
        print(f"Training completed in {epoch+1} epochs! Final loss = {loss.detach().item():.4f}")


    def predict(self, X_test):
        """
        Use the trained neural network to predict the labels of the test set 
        """
        # convert data into appropriate format for {torch}
        X_test_torch = torch.tensor(X_test, dtype=torch.float32)
        arg = self.forward(X_test_torch)
        return torch.argmax(arg,dim=1)