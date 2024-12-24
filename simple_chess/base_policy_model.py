import torch.nn as nn
import torch.optim as optim
import torch


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Example CNN structure - you'll want to replace this
        self.conv1 = nn.Conv2d(12, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3)

        # 4096 = 64*64 possible moves
        self.move_predictor = nn.Linear(256 * 6 * 6, 4096)

    def forward(self, board_position):
        features = torch.relu(self.conv1(board_position))
        features = torch.relu(self.conv2(features))
        features = torch.relu(self.conv3(features))
        features_flat = features.view(-1, 256 * 6 * 6)
        move_probabilities = torch.softmax(self.move_predictor(features_flat), dim=1)
        return move_probabilities
