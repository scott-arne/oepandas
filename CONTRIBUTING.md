# Contributing to OEPandas

Thank you for your interest in contributing to OEPandas! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project follows a Code of Conduct that all contributors are expected to adhere to. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## How to Contribute

### Reporting Bugs

Before creating a bug report, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected behavior** vs **actual behavior**
- **Code samples** or test cases demonstrating the issue
- **Environment details** (Python version, pandas version, OpenEye Toolkits version, OS)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- **Clear description** of the proposed feature
- **Use cases** explaining why this would be useful
- **Examples** of how the feature would work
- **Potential implementation approach** (if you have ideas)

### Pull Requests

1. **Fork the repository** and create your branch from `master`
2. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Make your changes**:
   - Write clear, documented code
   - Follow existing code style and conventions
   - Add tests for new functionality
   - Update documentation as needed

4. **Test your changes**:
   ```bash
   pytest
   ```

5. **Commit your changes**:
   - Use clear, descriptive commit messages
   - Reference relevant issue numbers

6. **Push to your fork** and submit a pull request

## Development Setup

### Prerequisites

- Python 3.10 or higher
- OpenEye Toolkits 2023.1.0 or higher
- Git

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/oepandas.git
cd oepandas

# Install in development mode
pip install -e ".[dev]"

# Run tests to verify setup
pytest
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=oepandas --cov-report=html

# Run specific test file
pytest tests/test_molecule.py

# Run specific test
pytest tests/test_molecule.py::test_create_simple
```

### Code Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use type hints for function signatures
- Write docstrings for public APIs
- Keep functions focused and well-documented

### Documentation

- Update README.md if adding new features
- Add docstrings to new functions and classes
- Include examples in docstrings when helpful
- Update example notebooks if relevant

## Project Structure

```
oepandas/
├── oepandas/              # Main package
│   ├── arrays/            # Extension array implementations
│   │   ├── base.py        # Base extension array
│   │   ├── molecule.py    # Molecule array
│   │   └── design_unit.py # Design unit array
│   ├── pandas_extensions.py  # DataFrame/Series accessors
│   ├── util.py            # Utility functions
│   └── exception.py       # Custom exceptions
├── tests/                 # Test suite
├── examples/              # Jupyter notebook examples
└── docs/                  # Documentation (if applicable)
```

## Testing Guidelines

- Write tests for new features
- Ensure tests pass before submitting PR
- Aim for high test coverage (>70%)
- Test edge cases and error handling
- Use descriptive test names

Example test structure:
```python
def test_feature_name():
    """Test description"""
    # Setup
    data = create_test_data()

    # Execute
    result = function_under_test(data)

    # Assert
    assert result == expected_value
```

## Review Process

1. **Automated checks** run on all pull requests (tests, linting)
2. **Code review** by maintainers
3. **Discussion** and iteration if needed
4. **Merge** once approved

## Questions?

- Open an issue for questions about contributing
- Check existing issues and pull requests for similar discussions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be recognized in the project. Thank you for helping make OEPandas better!
