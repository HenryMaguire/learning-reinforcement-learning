import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import chess
from simple_chess.base_policy_model import PolicyNetwork
from torch.utils.data import DataLoader, Dataset
import multiprocessing
import os
import pickle


def generate_random_board(max_moves=20):
    board = chess.Board()
    num_moves = np.random.randint(1, max_moves)
    for _ in range(num_moves):
        if board.is_game_over():
            break
        move = np.random.choice(list(board.legal_moves))
        board.push(move)

    # Ensure the final state is not game over
    while board.is_game_over():
        board.pop()

    return board


def board_to_state(board):
    state = np.zeros((12, 8, 8), dtype=np.float32)
    piece_idx = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5}

    for i in range(64):
        piece = board.piece_at(i)
        if piece is not None:
            color = int(piece.color)
            piece_type = piece_idx[piece.symbol().upper()]
            rank, file = i // 8, i % 8
            state[piece_type + 6 * color, rank, file] = 1

    return state


def generate_legal_action_mask(board):
    mask = np.zeros(4096, dtype=np.float32)
    for move in board.legal_moves:
        from_square = move.from_square
        to_square = move.to_square
        mask[from_square * 64 + to_square] = 1
    return mask


def generate_boards(num_boards, max_moves=20):
    unique_boards = set()
    boards = []
    legal_actions = []
    while len(unique_boards) < num_boards:
        board = generate_random_board(max_moves)
        board_fen = board.fen()
        if board_fen not in unique_boards:
            unique_boards.add(board_fen)
            boards.append(board_to_state(board))
            actions = generate_legal_action_mask(board)
            if np.sum(actions) > 0:
                legal_actions.append(actions)
    return np.stack(boards), np.stack(legal_actions)


def save_boards_to_disk(
    num_boards=10000,
    max_moves=100,
    filename="chess_boards.npy",
    actions_filename="chess_legal_actions.npy",
):
    with multiprocessing.Pool() as pool:
        results = pool.starmap(
            generate_boards,
            [(num_boards // multiprocessing.cpu_count(), max_moves)]
            * multiprocessing.cpu_count(),
        )
        boards, legal_actions = zip(*results)
        boards = np.vstack(boards)
        legal_actions = np.vstack(legal_actions)
        print(f"Saving {boards.shape[0]} boards to {filename}")
        np.save(filename, boards)
        print(f"Saving {legal_actions.shape[0]} legal actions to {actions_filename}")
        np.save(actions_filename, legal_actions)


class RandomChessDataset(Dataset):
    def __init__(
        self,
        boards_path="chess_boards.npy",
        legal_actions_path="chess_legal_actions.npy",
    ):
        self.boards = np.load(boards_path, mmap_mode="r")
        self.legal_actions = np.load(legal_actions_path, mmap_mode="r")
        self.num_samples = self.boards.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        norm = np.sum(self.legal_actions[idx])
        assert norm > 0, "No legal actions found"
        return (
            torch.tensor(self.boards[idx], dtype=torch.float32),
            torch.tensor(self.legal_actions[idx] / norm, dtype=torch.float32),
        )


def pretrain_policy_model():
    # Hyperparameters
    batch_size = 32
    learning_rate = 5e-3
    num_epochs = 10
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    # Initialize dataset and dataloader
    dataset = RandomChessDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = PolicyNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for states, target_probs in dataloader:
            states = states.to(device)
            target_probs = target_probs.to(device)

            outputs = model(states)
            loss = criterion(outputs, target_probs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

    # Save the pretrained model
    torch.save(model.state_dict(), "pretrained_policy.pt")


if __name__ == "__main__":
    if not os.path.exists("chess_boards.npy"):
        save_boards_to_disk()

    pretrain_policy_model()
