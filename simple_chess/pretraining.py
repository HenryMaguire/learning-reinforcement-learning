import random
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
from torch.utils.data import random_split


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


def board_from_state(state: np.ndarray) -> chess.Board:
    assert state.shape == (12, 8, 8)
    board = chess.Board()
    board.clear()  # Clear the board to set pieces manually
    piece_idx = {0: "P", 1: "N", 2: "B", 3: "R", 4: "Q", 5: "K"}

    for i in range(64):
        rank, file = i // 8, i % 8
        piece_channel = state[:, rank, file].argmax()
        if piece_channel < 6:
            piece = chess.Piece.from_symbol(piece_idx[piece_channel])
            board.set_piece_at(i, piece)
        else:
            piece = chess.Piece.from_symbol(piece_idx[piece_channel - 6].lower())
            board.set_piece_at(i, piece)

    return board


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
    num_boards=1000000,
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
        max_samples=1000000,
    ):
        self.boards = np.load(boards_path, mmap_mode="r")
        self.legal_actions = np.load(legal_actions_path, mmap_mode="r")
        self.num_samples = min(self.boards.shape[0], max_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        assert np.sum(self.legal_actions[idx]) > 0, "No legal actions found"
        return (
            torch.tensor(self.boards[idx], dtype=torch.float32),
            torch.tensor(self.legal_actions[idx], dtype=torch.float32),
        )


def pretrain_policy_model():
    # Hyperparameters
    batch_size = 32
    learning_rate = 5e-3
    num_epochs = 8
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    # Initialize dataset and dataloader
    dataset = RandomChessDataset(max_samples=200000)

    train_size = int(0.95 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Initialize model, loss function, and optimizer
    model = PolicyNetwork(with_softmax=False).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for states, target in train_loader:
            states = states.to(device)
            target = target.to(device)

            outputs = model(states)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            if norm > 2.0:
                print(f"Gradient norm: {norm}")
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}")

        test_loss = 0
        invalid_probs = []
        for states, target in test_loader:
            states = states.to(device)
            target = target.to(device)  # Valid mask.
            outputs = model(states)
            loss = criterion(outputs, target)
            test_loss += loss.item()
            pred_probs = torch.softmax(outputs, dim=1)
            invalid_prob = (pred_probs * (1 - target)).sum(dim=1).mean()
            invalid_probs.append(invalid_prob.item())

        print(f"Test Loss: {test_loss/len(test_loader)}")
        print(f"Invalid Prob: {np.mean(invalid_probs)}")
        scheduler.step()
        # Save the pretrained model
        torch.save(model.state_dict(), "pretrained_policy.pt")


if __name__ == "__main__":
    if not os.path.exists("chess_boards.npy"):
        save_boards_to_disk()

    pretrain_policy_model()
