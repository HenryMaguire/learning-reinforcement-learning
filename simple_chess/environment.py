import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import chess
import random
from collections import deque


class ChessEnvironment:
    def __init__(self, max_game_length=100):
        self.board = chess.Board()
        self.move_count = 0
        self.piece_values = {
            "P": 1,
            "N": 3,
            "B": 3,
            "R": 5,
            "Q": 9,
            "K": 0,  # King is not captured
        }
        self.max_game_length = max_game_length
        self.player_scores = {True: 0, False: 0}  # True for white, False for black

    def reset(self):
        self.board = chess.Board()
        self.move_count = 0
        return self._get_state()

    def _get_state(self):
        # Convert board to 8x8x12 binary tensor (6 piece types x 2 colors)
        state = np.zeros((8, 8, 12), dtype=np.float32)
        piece_idx = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5}

        for i in range(64):
            piece = self.board.piece_at(i)
            if piece is not None:
                color = int(piece.color)
                piece_type = piece_idx[piece.symbol().upper()]
                rank, file = i // 8, i % 8
                state[rank, file, piece_type + 6 * color] = 1

        return state

    def _get_legal_moves_mask(self):
        # Create a mask of legal moves (1 for legal, 0 for illegal)
        mask = np.zeros(4096, dtype=np.float32)  # 64*64 possible moves
        for move in self.board.legal_moves:
            from_square = move.from_square
            to_square = move.to_square
            mask[from_square * 64 + to_square] = 1
        return mask

    def step(self, action, result_weight: float = 0.5):
        # Convert action index to chess move
        from_square = action // 64
        to_square = action % 64
        move = chess.Move(from_square, to_square)

        # Check if move is legal
        if move not in self.board.legal_moves:
            return self._get_state(), -1, True, {}

        # Track captured pieces
        captured_piece_value = 0
        if move in self.board.move_stack:
            captured_piece = self.board.piece_at(move.to_square)
            if captured_piece is not None:
                captured_piece_value = self.piece_values[
                    captured_piece.symbol().upper()
                ]
                self.player_scores[
                    not self.board.turn
                ] += captured_piece_value  # Update opponent's score

        # Calculate score difference
        score_difference = self.player_scores[True] - self.player_scores[False]
        print(score_difference)
        # Initialize reward with score difference
        reward = score_difference
        result_reward = 0
        # Check game state
        done = self.board.is_game_over()
        if done:
            if self.board.is_checkmate():
                result_reward = 1 if self.board.turn else -1  # Win or loss
            elif self.board.is_stalemate() or self.board.is_insufficient_material():
                result_reward = 0  # Draw

        # Combine rewards
        total_reward = (1 - result_weight) * reward + result_weight * result_reward

        # Make move
        self.board.push(move)
        self.move_count += 1

        if self.move_count > 100:
            done = True

        return self._get_state(), total_reward, done, {}
