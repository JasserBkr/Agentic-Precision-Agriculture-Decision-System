import numpy as np

from agri_agent.data_access.satellite import (
    bbox_to_geojson_polygon,
    compute_ndvi,
)


def test_bbox_to_geojson_polygon_shape():
    bbox = {"min_lon": 10.0, "min_lat": 36.0, "max_lon": 10.1, "max_lat": 36.1}
    geom = bbox_to_geojson_polygon(bbox)
    assert geom["type"] == "Polygon"
    assert len(geom["coordinates"][0]) == 5  # closed ring


def test_compute_ndvi_known_values():
    red = np.array([0.1])
    nir = np.array([0.5])
    ndvi = compute_ndvi(red, nir)
    assert -1.0 <= ndvi[0] <= 1.0
    # (0.5-0.1)/(0.5+0.1) = 0.6667
    assert abs(ndvi[0] - 0.6667) < 1e-3
