import torch.nn as nn
import torch.optim as optim
import torch


class PolicyNetwork(nn.Module):
    def __init__(self, with_softmax: bool = False):
        super().__init__()
        # Example CNN structure - you'll want to replace this
        self.conv1 = nn.Conv2d(12, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3)
        # Without padding, the output is 6x6.
        self.linear1 = nn.Linear(256 * 6 * 6, 4096)
        # 4096 = 64*64 possible moves
        self.move_predictor = nn.Linear(4096, 4096)
        self.dropout = nn.Dropout(0.3)
        self.with_softmax = with_softmax

    def forward(self, board_position):
        features = torch.relu(self.conv1(board_position))
        features = torch.relu(self.conv2(features))
        features = torch.relu(self.conv3(features))
        features_flat = features.view(-1, 256 * 6 * 6)
        features_flat = self.dropout(features_flat)
        features_flat = torch.relu(self.linear1(features_flat))
        features_flat = self.dropout(features_flat)
        if self.with_softmax:
            move_probabilities = torch.softmax(
                self.move_predictor(features_flat), dim=1
            )
        else:
            move_probabilities = self.move_predictor(features_flat)
        return move_probabilities
