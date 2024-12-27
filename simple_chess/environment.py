import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import chess
import random
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class ChessEnv(gym.Env):
    def __init__(
        self,
        max_game_length=100,
        white_score_weight: float = 0.5,
        black_score_weight: float = 0.5,
    ):
        super(ChessEnv, self).__init__()
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
        self.white_score_weight = white_score_weight
        self.black_score_weight = black_score_weight
        self.checkmate_reward = 50
        self.loss_reward = -50
        self.check_reward = 0
        self.draw_reward = 0
        self.timestep_reward = -0.1
        self.illegal_move_reward = 0

        # Define action and observation space
        self.action_space = spaces.Discrete(4096)  # 64*64 possible moves
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(12, 8, 8), dtype=np.float32
        )

    def estimate_max_reward(self, num_checks=5):
        score = 0
        for piece in [("P", 8), ("N", 2), ("B", 2), ("R", 2), ("Q", 1)]:
            score += self.piece_values[piece[0]] * piece[1]
        # Taking all pieces yields 39 points.
        return self.checkmate_reward + self.check_reward * num_checks + score

    def reset(self, seed=None, return_info=False, options=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
        self.board = chess.Board()
        self.move_count = 0
        return self._get_state(), {}

    def _get_state(self):
        # Convert board to 8x8x12 binary tensor (6 piece types x 2 colors)
        state = np.zeros((12, 8, 8), dtype=np.float32)
        piece_idx = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5}

        for i in range(64):
            piece = self.board.piece_at(i)
            if piece is not None:
                color = int(piece.color)  # 0 for black, 1 for white
                piece_type = piece_idx[piece.symbol().upper()]
                rank, file = i // 8, i % 8
                state[piece_type + 6 * color, rank, file] = 1

        return state

    def _get_legal_moves_mask(self):
        mask = np.zeros(4096, dtype=np.float32)  # 64*64 possible moves
        for move in self.board.legal_moves:
            from_square = move.from_square
            to_square = move.to_square
            mask[from_square * 64 + to_square] = 1
        return mask

    def _calculate_positional_reward(self, is_white_turn: bool):
        # You can't move into check, so only check opposite king.
        king_square = self.board.king(not is_white_turn)
        if king_square is not None:
            attackers = len(self.board.attackers(is_white_turn, king_square))
            # Being in check two ways is polynomially worse than one
            reward = self.check_reward * attackers**2
            # if is_white_turn, black is in check
            return reward if is_white_turn else -reward
        return 0

    def _calculate_capture_points(self, move):
        captured_piece = self.board.piece_at(move.to_square)
        if captured_piece is not None:
            return self.piece_values[captured_piece.symbol().upper()]
        else:
            return 0

    def _calculate_rewards(self, is_white_turn: bool, capture_points: int):
        capture_rewards = capture_points
        if not is_white_turn:
            capture_rewards = -capture_rewards

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

        positional_reward = self._calculate_positional_reward(
            is_white_turn=is_white_turn
        )
        if is_white_turn:
            total_reward = (
                self.white_score_weight * capture_rewards
                + positional_reward
                + result_reward
            )
        else:
            total_reward = (
                self.black_score_weight * capture_rewards
                + positional_reward
                + result_reward
            )
        return total_reward, done

    def get_random_action(self, as_index: bool = False):
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
        truncated = False  # TODO(hm): use truncation.
        # Check if move is legal
        if move not in self.board.legal_moves:
            self.move_count += 1
            done = True if self.move_count > self.max_game_length else False
            return self._get_state(), self.illegal_move_reward, done, truncated, {}

        capture_points = self._calculate_capture_points(move)
        self.board.push(move)
        self.move_count += 1

        player_reward, done = self._calculate_rewards(
            is_white_turn=True, capture_points=capture_points
        )
        if done or self.move_count > self.max_game_length:
            return self._get_state(), player_reward, True, truncated, {}

        assert not self.board.turn
        ai_action = self.get_random_action()
        capture_points = self._calculate_capture_points(ai_action)
        self.board.push(ai_action)
        ai_reward, done = self._calculate_rewards(
            is_white_turn=False, capture_points=capture_points
        )
        reward = player_reward + ai_reward
        if done:
            return self._get_state(), reward, done, truncated, {}

        reward += self.timestep_reward
        return self._get_state(), reward, done, truncated, {}

    def render(self):
        print(self.board)
