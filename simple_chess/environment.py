import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import chess
import random
from collections import deque


class ChessEnvironment:
    def __init__(self, max_game_length=100, score_weight: float = 0.5):
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
        self.score_weight = score_weight
        self.checkmate_reward = 200
        self.loss_reward = -200
        self.draw_reward = 0
        self.timestep_reward = 0
        self.illegal_move_reward = -1

    def reset(self):
        self.board = chess.Board()
        self.move_count = 0
        return self._get_state()

    def _get_state(self):
        # Convert board to 8x8x12 binary tensor (6 piece types x 2 colors)
        state = np.zeros((12, 8, 8), dtype=np.float32)
        piece_idx = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5}

        for i in range(64):
            piece = self.board.piece_at(i)
            if piece is not None:
                color = int(piece.color)
                piece_type = piece_idx[piece.symbol().upper()]
                rank, file = i // 8, i % 8
                state[piece_type + 6 * color, rank, file] = 1

        return state

    def _get_legal_moves_mask(self):
        # Create a mask of legal moves (1 for legal, 0 for illegal)
        mask = np.zeros(4096, dtype=np.float32)  # 64*64 possible moves
        for move in self.board.legal_moves:
            from_square = move.from_square
            to_square = move.to_square
            mask[from_square * 64 + to_square] = 1
        return mask

    def _calculate_rewards(
        self,
        move,
        is_white_turn: bool,
    ):
        # Track captured pieces
        captured_piece_value = 0
        # Check if the move captures a piece
        if move in self.board.move_stack:
            captured_piece = self.board.piece_at(move.to_square)
            if captured_piece is not None:
                captured_piece_value = self.piece_values[
                    captured_piece.symbol().upper()
                ]
        if not is_white_turn:
            captured_piece_value = -captured_piece_value

        result_reward = 0
        done = self.board.is_game_over()
        if done:
            if self.board.is_checkmate():
                result_reward = (
                    self.checkmate_reward if is_white_turn else self.loss_reward
                )
                if is_white_turn:
                    print("White wins")
                else:
                    print("Black wins")

            elif self.board.is_stalemate() or self.board.is_insufficient_material():
                print("Draw")
                result_reward = self.draw_reward
            else:
                print("Game over by other reason")

        total_reward = self.score_weight * captured_piece_value + result_reward
        return total_reward, done

    def _random_action(self, as_index: bool = False):
        move = random.choice(list(self.board.legal_moves))
        if as_index:
            return move.from_square * 64 + move.to_square
        return move

    def step(self, action):
        assert self.board.turn
        # Convert action index to chess move
        from_square = action // 64
        to_square = action % 64
        move = chess.Move(from_square, to_square)
        done = False
        # Check if move is legal
        if move not in self.board.legal_moves:
            self.move_count += 1
            done = True if self.move_count > self.max_game_length else False
            return self._get_state(), self.illegal_move_reward, done, {}

        # Make move
        self.board.push(move)
        self.move_count += 1

        player_reward, done = self._calculate_rewards(move, is_white_turn=True)
        if done or self.move_count > self.max_game_length:
            return self._get_state(), player_reward, True, {}

        assert not self.board.turn
        ai_action = self._random_action()
        self.board.push(ai_action)

        ai_reward, done = self._calculate_rewards(ai_action, is_white_turn=False)
        reward = player_reward + ai_reward
        if done:
            return self._get_state(), reward, True, {}

        reward += self.timestep_reward
        return self._get_state(), reward, done, {}

    def render(self):
        print(self.board)
