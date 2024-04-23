import torch
import torch.nn as nn
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_sequence):
        batch_size = input_sequence.size(0)
        hidden = self.initHidden(batch_size)
        output, _ = self.rnn(input_sequence, hidden)
        output = self.fc(output[:, -1, :])
        output = self.softmax(output)
        return output

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)





# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(RNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
#         self.i2o = nn.Linear(input_size + hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)
#
#     def forward(self, input, hidden):
#         combined = torch.cat((input, hidden), 1)
#         hidden = self.i2h(combined)
#         output = self.i2o(combined)
#         output = self.softmax(output)
#         return output, hidden
#
#     def initHidden(self):
#         return torch.zeros(1, self.hidden_size)

