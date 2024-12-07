.PHONY: test test-verbose clean

# Default Python interpreter
PYTHON = python

# Test settings
PYTEST = pytest
PYTEST_FLAGS = -v
TEST_DIR = tests/

# Main test command
test:
	$(PYTEST) $(TEST_DIR)

# Verbose test output
test-verbose:
	$(PYTEST) $(PYTEST_FLAGS) $(TEST_DIR)

# Clean up Python cache files
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -r {} + 