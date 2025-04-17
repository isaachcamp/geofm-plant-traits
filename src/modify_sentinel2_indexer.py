
from functools import cached_property

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import geoarrow.pyarrow as ga
import rasterio
from rasterio.enums import Resampling
from shapely.geometry import box
from shapely import Polygon
from stacchip.indexer import Sentinel2Indexer


class ModSentinel2Indexer(Sentinel2Indexer):
    """
    Modified version of the Sentinel2Indexer class – uses href property
    from Copernicus STAC assets to load scene classification (SCL) band data.
    """

    # @cached_property
    # def scl(self):
    #     """
    #     The Scene Classification (SCL) band data for the STAC item
    #     """
    #     print("Loading scl band")
    #     with rasterio.open(self.item.assets["SCL_20m"].href) as src:
    #         return src.read(out_shape=(1, *self.shape), resampling=Resampling.nearest)[
    #             0
    #         ]

    def get_vegetation_stats(self, x, y):
        """
        Get vegetation percentage for a chip using SCL band.
        """
        scl = self.scl[
            y * self.chip_size : (y + 1) * self.chip_size,
            x * self.chip_size : (x + 1) * self.chip_size,
        ]

        vegetated_percentage = np.sum(scl == 4) / scl.size

        return vegetated_percentage

    def filter_by_vegetation_cover(self, index: pa.Table) -> pa.Table:
        """
        The index for this STAC item
        """

        veg_perc_col = np.empty(index.shape[0], dtype="float32")

        for i in range(0, index.shape[0]):
            x = index["chip_index_x"].to_numpy()[i]
            y = index["chip_index_y"].to_numpy()[i]
            veg_perc_col[i] = self.get_vegetation_stats(x, y)

        index = index.append_column(
            "vegetated_percentage",
            pa.array(veg_perc_col),
        )
        chips_count = index.shape[0]
        index = index.filter(
            pc.field("vegetated_percentage") == 1.
        )
        print(
            f"Dropped {chips_count - index.shape[0]}/{chips_count} chips due to no vegetation cover"
        )
        return index

    def _get_single_pixel(self, x: int, y: int) -> Polygon:
        """Bounding box for a single pixel"""
        # Get the top left corner of the central pixel.
        tl_x = self.bbox[0] + x * self.transform[0] * self.chip_size
        tl_y = self.bbox[3] + y * self.transform[4] * self.chip_size

        center_tl_x = np.float64(tl_x +  self.transform[0] * (self.chip_size // 2))
        center_tl_y = np.float64(tl_y +  self.transform[4] * (self.chip_size // 2))

        # Randomly sample from pixels near center
        # e.g., if chip size is 9, then sample from surrounding 2 pixels.
        # Avoids structured sampling, possibly introducing bias.
        offset = self.chip_size // 4
        shift_x, shift_y = np.random.randint(-offset, offset+1, size=2)

        # Get the bounding box for the pixel
        pixel_box = box(
            minx=center_tl_x + shift_x * self.transform[0],
            miny=center_tl_y + shift_y * self.transform[4],
            maxx=center_tl_x + (shift_x + 1) * self.transform[0],
            maxy=center_tl_y + (shift_y + 1) * self.transform[4],
        )

        pixel_x = x * self.chip_size + shift_x
        pixel_y = y * self.chip_size + shift_y

        print(pixel_x, pixel_y)

        return self.reproject(pixel_box).wkt, (pixel_x, pixel_y)

    def get_single_pixel(self, index: pa.Table) -> pa.Table:
        """
        The index for this STAC item
        """

        pixels = np.empty(index.shape[0], dtype="object")
        pixels_x = np.empty(index.shape[0], dtype="int16")
        pixels_y = np.empty(index.shape[0], dtype="int16")

        for i in range(0, index.shape[0]):
            x = int(index["chip_index_x"].to_numpy()[i])
            y = int(index["chip_index_y"].to_numpy()[i])
            pixels[i], (pixels_x[i], pixels_y[i]) = self._get_single_pixel(x, y)

        pixels = ga.as_geoarrow(pixels)

        index = index.append_column("x_pixel", pa.array(pixels_x))
        index = index.append_column("y_pixel", pa.array(pixels_y))
        index = index.append_column("pixel_geometry", pixels)

        return index
