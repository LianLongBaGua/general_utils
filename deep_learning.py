import torch.nn as nn
import torch
import torch.optim


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn1 = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0_1 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)  # initial hidden state
        out, _ = self.rnn1(x, h0_1)  # out: [batch, seq, hidden_size]
        out = torch.relu(out)
        out = self.fc(out[:, -1, :])  # decode the hidden state of the last time step
        return out


def define_model(model: nn.Module, criterion: nn.Module, input_size: int, hidden_size: int, output_size: int, lr=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Instantiate the model, define loss function and optimizer
    model = RNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer, device


def train_eval_model(model: nn.Module, criterion: nn.Module, optimizer: torch.optim, train_loader, test_loader, epochs: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(epochs):
        model.train()
        for i, (inputs, targets) in enumerate(train_loader, 0):
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Forward pass to get output/logits
            outputs = model(inputs)
            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, targets)
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch: {epoch + 1}, loss: {loss.item():.4f}')
            # evaluate performance on test set
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(test_loader, 0):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    # Get the indices of the maxima
                    max_indices = outputs.argmax(dim=1)
                    # Create a tensor filled with zeros, with the same shape as outputs
                    one_hot = torch.zeros_like(outputs)
                    # Use scatter_ to set the indices of the maxima in each row to 1
                    one_hot.scatter_(1, max_indices.unsqueeze(1), 1)
                    identical_elements = (one_hot == targets).all(dim=1)
                    correct += identical_elements.sum()
            total = len(test_loader.dataset)
            print(f'Accuracy of the network on the test set: \
                  {(correct) / total * 100}%')
    return model
