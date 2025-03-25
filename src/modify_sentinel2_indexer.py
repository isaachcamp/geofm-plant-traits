
from functools import cached_property

import rasterio
from rasterio.enums import Resampling
from stacchip.indexer import Sentinel2Indexer


class ModSentinel2Indexer(Sentinel2Indexer):
    """
    Modified version of the Sentinel2Indexer class – uses href property
    from Copernicus STAC assets to load scene classification (SCL) band data.
    """

    @cached_property
    def scl(self):
        """
        The Scene Classification (SCL) band data for the STAC item
        """
        print("Loading scl band")
        with rasterio.open(self.item.assets["SCL_20m"].href) as src:
            return src.read(out_shape=(1, *self.shape), resampling=Resampling.nearest)[
                0
            ]
