
from pathlib import Path

from pystac import Item

from src.modify_sentinel2_indexer import ModSentinel2Indexer

def load_indexer(ipath: Path, platform: str, item_id: str) -> ModSentinel2Indexer:
    """Load stacchip index table from local file"""
    item = Item.from_file(ipath / Path(f"{platform}/{item_id}/stac_item.json"))
    return ModSentinel2Indexer(item, chip_size=1)

