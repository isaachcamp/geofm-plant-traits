
import pytest
from pystac import Item
import pandas as pd


class TestDropDuplicateItems:
    item_collection = [
            Item(
                id="1",
                geometry=None,
                bbox=None,
                datetime=pd.to_datetime("2021-01-01"),
                properties={"name": "item1"}
            ),
            Item(
                id="2",
                geometry=None,
                bbox=None,
                datetime=pd.to_datetime("2020-01-01"),
                properties={"name": "item2"}
            ),
            Item(
                id="1",
                geometry=None,
                bbox=None,
                datetime=pd.to_datetime("2021-01-01"),
                properties={"name": "item1"}
            ),
        ]

    def test_correct_ids_kept(self):
        from src.stac_filter_utils import drop_duplicate_items

        unique_items = drop_duplicate_items(self.item_collection)

        assert len(unique_items) == 2
        assert set(item.id for item in unique_items) == set(["1", "2"])

    def test_correct_types(self):
        from src.stac_filter_utils import drop_duplicate_items

        unique_items = drop_duplicate_items(self.item_collection)

        assert isinstance(unique_items, list)
        assert isinstance(unique_items[0], Item)
