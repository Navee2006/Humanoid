"""
Pytest configuration and fixtures for web agent tests.
"""
import pytest
from pathlib import Path
import tempfile
import shutil
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup after test
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_html():
    """Provide sample HTML content for testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Test Page</title></head>
    <body>
        <h1>Test Heading</h1>
        <p>Test paragraph</p>
    </body>
    </html>
    """
