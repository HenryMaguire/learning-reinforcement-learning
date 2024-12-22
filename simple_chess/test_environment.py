import pytest

from simple_chess.environment import ChessEnvironment


@pytest.fixture
def chess_env():
    return ChessEnvironment()


def test_reset(chess_env):
    state = chess_env.reset()
    assert chess_env.move_count == 0
    assert chess_env.board.is_valid()


def test_get_state(chess_env):
    chess_env.reset()
    state = chess_env._get_state()
    assert state.shape == (8, 8, 12)  # Check state shape


def test_get_legal_moves_mask(chess_env):
    chess_env.reset()
    mask = chess_env._get_legal_moves_mask()
    assert mask.shape == (4096,)  # Check mask shape
    assert mask.sum() > 0  # Ensure there are legal moves


def test_step_legal_move(chess_env):
    chess_env.reset()
    legal_moves = list(chess_env.board.legal_moves)  # Get legal moves
    assert legal_moves, "No legal moves available"  # Ensure there are legal moves
    action = (
        legal_moves[0].from_square * 64 + legal_moves[0].to_square
    )  # Convert to action index
    state, reward, done, _ = chess_env.step(action)
    assert not done  # Game should not be over
    assert chess_env.move_count == 1  # Move count should increment


def test_step_illegal_move(chess_env):
    chess_env.reset()
    action = 100  # Invalid action
    piece_values = chess_env.piece_values
    state, reward, done, _ = chess_env.step(action, piece_values)
    assert reward == -1  # Illegal move should return -1
