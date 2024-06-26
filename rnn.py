import torch
import torch.nn as nn
import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, sequence_length=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.sequence_length = sequence_length
        self.output_size = output_size
        self.rnn = nn.RNN(input_size, hidden_size,num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size * self.sequence_length, self.output_size)


    def forward(self, input_sequence):
        batch_size = input_sequence.size(0)
        inputs = input_sequence.unfold(dimension = 0,size = self.sequence_length, step = 1) # 2
        h0 = torch.ones(self.num_layers , batch_size, self.hidden_size)

        print("batch_size:", batch_size)
        print("input_sequence:", input_sequence.size())
        hidden = self.initHidden(batch_size)
        print("hidden:", hidden.size())
        # output, _ = self.rnn(input_sequence, hidden)

        # output = self.fc(output[:, -1, :])
        out, _ = self.rnn(inputs, h0)
        out = out.reshape(batch_size, self.hidden_size * self.sequence_length)  # 5
        out = self.fc(out)
        log_probs = nn.functional.log_softmax(out, dim=1)
        # output = self.fc(output.squeeze(1))
        # log_probs = nn.functional.log_softmax(output, dim=1)
        return log_probs

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(RNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, input_sequence):
#         batch_size = input_sequence.size(0)
#         hidden = self.initHidden(batch_size)
#         output, _ = self.rnn(input_sequence, hidden)
#         output = self.fc(output[:, -1, :])
#         log_probs = nn.functional.log_softmax(output, dim=1)
#         return log_probs
#
#     def initHidden(self, batch_size):
#         return torch.zeros(1, self.hidden_size)



# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(RNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, input_sequence):
#         batch_size = input_sequence.size(0)
#         hidden = self.initHidden(batch_size)
#         output, _ = self.rnn(input_sequence, hidden)
#         output = self.fc(output[:, -1, :])
#         log_probs = nn.functional.log_softmax(output, dim=1)
#         return log_probs
#
#     def initHidden(self, batch_size):
#         return torch.zeros(1, batch_size, self.hidden_size)





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

