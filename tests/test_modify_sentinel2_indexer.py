
import pytest
import numpy as np
from pystac import Item

from src.modify_sentinel2_indexer import ModSentinel2Indexer

class TestGetSinglePixelDunder:
    """
    Test the get_single_pixel_dunder method of the ModSentinel2Indexer class.
    """
    @pytest.fixture
    def mock_randint(self, monkeypatch):
        """Fixture that allows setting specific return values for randint."""
        return_pairs = []

        def mock_randint_fn(low, high=None, size=None, dtype=int):
            if not return_pairs:
                raise ValueError("No return values set for np.random.randint")

            if size == 2:
                return np.array(return_pairs)

        # Replace np.random.randint with our mock
        monkeypatch.setattr(np.random, "randint", mock_randint_fn)

        def set_return_pairs(*pairs):
            """Set pairs of values to return for shift_x, shift_y"""
            nonlocal return_pairs
            return_pairs = list(pairs)

        return set_return_pairs

    def test_negative_xy(self, mock_randint):
        """
        Checks that the method returns the expected pixel coordinates
        when given x and y coordinates with negative shifts.
        """
        mock_randint(-2, -1) # Set randint output.

        item = Item.from_file(
            "tests/data/sentinel-2-l2a-S2A_T20HNJ_20240311T140636_L2A.json"
        )

        chip_size = 10
        indexer = ModSentinel2Indexer(item, chip_size)
        indexer.shape = (10, 10)
        x, y = 5, 5

        shift_x, shift_y = [-2, -1]
        _, result = indexer._get_single_pixel(x, y)
        expected_result = (50 + shift_x, 50 + shift_y) # (48, 49)

        assert result == expected_result

    def test_zero_xy(self, mock_randint):
        """
        Checks that the method returns the expected pixel coordinates
        when given specific x and y coordinates.
        """
        mock_randint(0, 0) # Set randint output.

        item = Item.from_file(
            "tests/data/sentinel-2-l2a-S2A_T20HNJ_20240311T140636_L2A.json"
        )

        chip_size = 10  # Mock chip size
        indexer = ModSentinel2Indexer(item, chip_size)  # Mock item
        indexer.shape = (10, 10)  # Mock shape

        x, y = 5, 5
        _, result = indexer._get_single_pixel(x, y)

        expected_result = (50, 50)
        assert result == expected_result

    def test_positive_xy(self, mock_randint):
        """
        Checks that the method returns the expected pixel coordinates
        when given specific x and y coordinates.
        """
        mock_randint(1, 2) # Set randint output.

        item = Item.from_file(
            "tests/data/sentinel-2-l2a-S2A_T20HNJ_20240311T140636_L2A.json"
        )

        chip_size = 10  # Mock chip size
        indexer = ModSentinel2Indexer(item, chip_size)  # Mock item
        indexer.shape = (10, 10)  # Mock shape

        x, y = 5, 5
        shift_x, shift_y = [1, 2]
        _, result = indexer._get_single_pixel(x, y)

        expected_result = (50 + shift_x, 50 + shift_y) # (51, 52)
        assert result == expected_result
    