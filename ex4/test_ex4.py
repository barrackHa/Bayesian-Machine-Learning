import pytest
from ex4 import *

from pathlib import Path

p = Path(__file__)

def test():
    assert True

if __name__ == '__main__':
    pytest.main([str(p)])

