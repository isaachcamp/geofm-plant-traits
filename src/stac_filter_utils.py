
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from shapely.geometry import shape
import geojson
from pystac import Item


def _create_spatial_intersection_dict(feature: Dict[str, Any]):
    """Create a spatial intersection filter dictionary for a GeoJSON feature."""
    if feature['geometry']['type'] != 'Polygon':
        raise TypeError('Feature must be a Polygon shape.')
    return {
        "op": "s_intersects",
        "args": [
          { "property": "geometry" } ,
          feature['geometry']
        ]
      }

def create_spatial_intersection_dict(feature_collection: Dict[str, Any]):
    """Create a spatial intersection filter dictionary for a GeoJSON FeatureCollection."""
    # Handles spatial intersection with multiple polygons.
    return {
        "filter": {
            "op": "or" ,
            "args": [_create_spatial_intersection_dict(f) for f in feature_collection['features']]
        }
    }

def create_temporal_intersection_dict(feature: Dict[str, Any]):
    """Create a temporal intersection filter dictionary."""
    filter_dict = {"op": "or", "args": []}

    if not 'datetime' in feature['properties']:
        raise KeyError('Plot must have a datetime fro data collection.')

    datetime = feature['properties']['datetime']
    year, month, _ = datetime.split('-')

    year_range = np.arange(int(year) - 2, int(year) + 3)
    start_month, end_month = int(month) - 1, int(month) + 1

    for year in year_range:
        filter_dict['args'].append({
          "op": "t_intersects",
          "args": [
            {"property": "datetime" },
            {"interval": [ f"{year}-{start_month}-01T00:00:00", f"{year}-{end_month}-01T00:00:00"] }
          ]
        })

    return {"filter": filter_dict}

def drop_duplicate_items(item_collection: List[Item]) -> List[Item]:
    """Drop duplicate Items from list."""
    return list({item.id: item for item in item_collection}.values())

def find_nearest_timestamp_idx(item_timestamps: List[str], collection_timestamp: str) -> str:
    """Find index of the nearest timestamp to the data collection timestamp."""
    item_timestamps = [pd.to_datetime(ts) for ts in item_timestamps] # convert to datetime
    collection_timestamp = pd.to_datetime(collection_timestamp) # convert to datetime
    nearest_timestamp = min(item_timestamps, key=lambda x: abs(x - collection_timestamp))
    return item_timestamps.index(nearest_timestamp)

def select_nearest_timestamp(items: List[Item], collection_timestamp: str) -> Item:
    """Select the item with the nearest timestamp to the data collection timestamp."""
    timestamps = [item.properties['datetime'] for item in items]
    idx = find_nearest_timestamp_idx(timestamps, collection_timestamp)
    return items[idx]

def intersection_percent(chip: pd.Series, aoi: geojson.Polygon) -> float:
    """Calculate percentage area of a chip that intersects the plot."""
    if 'geometry' not in chip:
        raise KeyError('Chip must have associated geometry.')
    geom_item = shape(chip.geometry)
    geom_aoi = shape(aoi)

    intersected_geom = geom_aoi.intersection(geom_item)
    return (intersected_geom.area * 100) / geom_aoi.area
