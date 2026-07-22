"""
Sentinel-2 access and vegetation/water index computation via Google
Earth Engine. Week 1 deliverable.
 
Computes the three indices mandated by the project brief — NDVI, EVI,
NDWI — plus GNDVI and SAVI as useful supplementary indices for later
weeks (not required by the brief, kept clearly separate below).
 
Requires agri_agent.utils.auth.init_earth_engine() to have been called
once beforehand (handles ee.Initialize with your registered Cloud
project) and, before that, one-time local authentication via
`ee.Authenticate()`.
"""
 
from datetime import date

import numpy as np
import ee
 
from agri_agent.utils.logging_config import get_logger
 
log = get_logger(__name__)
 
# Harmonized Sentinel-2 Surface Reflectance (L2A) collection.
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
 
# Indices required by the project brief.
REQUIRED_INDEX_BANDS = ["NDVI", "EVI", "NDWI"]
# Extra indices computed for convenience — not mandated, safe to ignore.
OPTIONAL_INDEX_BANDS = ["GNDVI", "SAVI"]
ALL_INDEX_BANDS = REQUIRED_INDEX_BANDS + OPTIONAL_INDEX_BANDS
 
 
def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute NDVI from red and NIR reflectance arrays."""
    return (nir - red) / (nir + red)


def bbox_to_geojson_polygon(bbox: dict) -> dict:
    """Convert a {min_lon, min_lat, max_lon, max_lat} dict to a GeoJSON polygon."""
    min_lon, min_lat = bbox["min_lon"], bbox["min_lat"]
    max_lon, max_lat = bbox["max_lon"], bbox["max_lat"]
    return {
        "type": "Polygon",
        "coordinates": [[
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat],
        ]],
    }
 
 
def bbox_to_ee_geometry(bbox: dict) -> ee.Geometry:
    """Convert a {min_lon, min_lat, max_lon, max_lat} dict to an ee.Geometry.Rectangle."""
    return ee.Geometry.Rectangle([
        bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"],
    ])
 
 
def mask_s2_clouds(image: ee.Image, cloud_prob_threshold: int = 20) -> ee.Image:
    """
    Mask clouds, cloud shadow, cirrus, and snow using the Scene
    Classification Layer (SCL) band, PLUS a per-pixel cloud probability
    threshold from the MSK_CLDPRB band. SCL alone is a coarse
    classification and frequently misses thin/edge clouds and haze,
    which can otherwise leak into "clear" scenes and distort indices
    (e.g. NDVI moving sharply while EVI/SAVI move the opposite direction
    over a few days is a classic sign of exactly this contamination).
 
    SCL class codes excluded: 3 = cloud shadow, 8/9 = cloud medium/high
    probability, 10 = thin cirrus, 11 = snow/ice.
    MSK_CLDPRB is a 0-100 per-pixel cloud probability; pixels at or
    above `cloud_prob_threshold` are masked out in addition to the SCL
    classes above.
    """
    scl = image.select("SCL")
    scl_mask = (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
 
    cloud_prob = image.select("MSK_CLDPRB")
    prob_mask = cloud_prob.lt(cloud_prob_threshold)
 
    return image.updateMask(scl_mask.And(prob_mask))
 
 
def add_indices(image: ee.Image) -> ee.Image:
    """
    Add NDVI, EVI, NDWI (required), and GNDVI, SAVI (optional, bonus)
    as new bands on the image.
 
    Sentinel-2 band reference: B2=blue, B3=green, B4=red, B8=NIR,
    B11=SWIR1. S2_SR_HARMONIZED reflectance is scaled by 10000, so we
    rescale to 0-1 before computing ratios.
    """
    scaled = image.divide(10000)
 
    ndvi = scaled.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = scaled.normalizedDifference(["B8", "B11"]).rename("NDWI")
    gndvi = scaled.normalizedDifference(["B8", "B3"]).rename("GNDVI")
 
    evi = scaled.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {
            "NIR": scaled.select("B8"),
            "RED": scaled.select("B4"),
            "BLUE": scaled.select("B2"),
        },
    ).rename("EVI")
 
    savi = scaled.expression(
        "((NIR - RED) / (NIR + RED + 0.5)) * 1.5",
        {"NIR": scaled.select("B8"), "RED": scaled.select("B4")},
    ).rename("SAVI")
 
    return image.addBands([ndvi, evi, ndwi, gndvi, savi])
 
 
def get_sentinel2_collection(
    bbox: dict,
    start_date: date,
    end_date: date,
    max_cloud_cover_pct: int = 20,
    cloud_prob_threshold: int = 20,
) -> ee.ImageCollection:
    """
    Return a cloud-masked, index-augmented Sentinel-2 collection over a
    field's bounding box and date range.
 
    `max_cloud_cover_pct` filters whole scenes by their SCENE-WIDE cloud
    percentage; `cloud_prob_threshold` additionally masks individual
    pixels within an otherwise-accepted scene (see mask_s2_clouds) —
    a scene can pass the scene-wide filter while still having a
    contaminated patch directly over your field.
    """
    geom = bbox_to_ee_geometry(bbox)
    collection = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(geom)
        .filterDate(start_date.isoformat(), end_date.isoformat())
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_cover_pct))
        .map(lambda img: mask_s2_clouds(img, cloud_prob_threshold))
        .map(add_indices)
    )
    return collection
 
 
def _expected_pixel_count(bbox: dict, scale: int = 10) -> int:
    """
    Compute the theoretical full-coverage pixel count for a bbox at a
    given resolution, from the geometry's actual geodesic area — used
    as the reference to detect scenes that only partially cover the
    field (see get_field_index_timeseries' min_valid_pixel_fraction).
    """
    geom = bbox_to_ee_geometry(bbox)
    area_m2 = geom.area(maxError=1).getInfo()
    return round(area_m2 / (scale * scale))
 
 
def get_field_index_timeseries(
    bbox: dict,
    start_date: date,
    end_date: date,
    max_cloud_cover_pct: int = 20,
    scale: int = 10,
    min_valid_pixel_fraction: float = 0.9,
) -> list[dict]:
    """
    Compute the field-mean value of every index (NDVI, EVI, NDWI, GNDVI,
    SAVI) for every available scene in the date range. This is the main
    Week 1 deliverable call — one row per scene date.
 
    `scale` is the reduction resolution in meters; 10 matches Sentinel-2's
    native resolution for the visible/NIR bands used here.
 
    `min_valid_pixel_fraction`: scenes where fewer than this fraction of
    the field's expected pixels are valid get DROPPED, not averaged.
    A partial-coverage scene (tile/swath edge cutting across the field)
    silently changes what "field mean" means for that date — it's not
    comparable to a full-field average on other dates, even with zero
    cloud contamination. This was discovered empirically: a scene with
    negligible cloud probability and normal sun geometry still produced
    a wildly divergent NDVI because only ~47% of the field was covered.
    """
    collection = get_sentinel2_collection(
        bbox, start_date, end_date, max_cloud_cover_pct
    )
    geom = bbox_to_ee_geometry(bbox)
    expected_count = _expected_pixel_count(bbox, scale)
 
    def reduce_image(image: ee.Image) -> ee.Feature:
        stats = image.select(ALL_INDEX_BANDS).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geom, scale=scale, maxPixels=1e9,
        )
        count = image.select("NDVI").reduceRegion(
            reducer=ee.Reducer.count(), geometry=geom, scale=scale, maxPixels=1e9,
        ).get("NDVI")
        return ee.Feature(None, stats.set("date", image.date().format("YYYY-MM-dd"))
                          .set("valid_pixel_count", count))
 
    features = collection.map(reduce_image)
    result = features.getInfo()
    all_records = [f["properties"] for f in result["features"]]
 
    kept, dropped = [], []
    for r in all_records:
        fraction = (r.get("valid_pixel_count") or 0) / expected_count
        if fraction >= min_valid_pixel_fraction:
            kept.append(r)
        else:
            dropped.append((r["date"], fraction))
 
    if dropped:
        log.warning(
            "Dropped %d/%d scenes for incomplete field coverage (< %.0f%% valid pixels): %s",
            len(dropped), len(all_records), min_valid_pixel_fraction * 100, dropped,
        )
 
    log.info(
        "Computed index time series for bbox %s: %d scenes kept, %d dropped (%s to %s)",
        bbox, len(kept), len(dropped), start_date, end_date,
    )
    return kept
 
 
def get_latest_field_indices(
    bbox: dict, lookback_days: int = 30, max_cloud_cover_pct: int = 20,
) -> dict | None:
    """
    Convenience wrapper: return just the most recent available scene's
    index values for a field, or None if nothing is found in the
    lookback window.
    """
    from datetime import timedelta
 
    end = date.today()
    start = end - timedelta(days=lookback_days)
    records = get_field_index_timeseries(bbox, start, end, max_cloud_cover_pct)
    if not records:
        log.warning("No scenes found in the last %d days for bbox %s", lookback_days, bbox)
        return None
    return sorted(records, key=lambda r: r["date"])[-1]
 
 
def debug_scene_diagnostics(
    bbox: dict,
    start_date: date,
    end_date: date,
    max_cloud_cover_pct: int = 20,
    cloud_prob_threshold: int = 20,
    scale: int = 10,
) -> list[dict]:
    """
    Diagnostic helper for investigating unexplained index jumps between
    scenes. For each scene, returns:
    - valid_pixel_count for NDVI: how many pixels actually fed the
      reduceRegion mean — a shrinking count across dates points to a
      changing valid-pixel footprint (tile edge clipping, partial
      masking) rather than real vegetation change.
    - mean_cloud_probability over the field, from MSK_CLDPRB directly
      (not thresholded) — confirms whether cloud probability was
      genuinely near zero, or just under the mask threshold.
    - spacecraft, relative_orbit, sun/view angles — different orbits or
      geometries between two dates can produce BRDF-driven index shifts
      with a perfectly clear sky, no cloud involved at all.
    """
    collection = get_sentinel2_collection(
        bbox, start_date, end_date, max_cloud_cover_pct, cloud_prob_threshold
    )
    geom = bbox_to_ee_geometry(bbox)
 
    def diagnose(image: ee.Image) -> ee.Feature:
        ndvi_stats = image.select("NDVI").reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
            geometry=geom, scale=scale, maxPixels=1e9,
        )
        cloud_prob_mean = image.select("MSK_CLDPRB").reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geom, scale=scale, maxPixels=1e9,
        ).get("MSK_CLDPRB")
 
        props = {
            "date": image.date().format("YYYY-MM-dd"),
            "ndvi_mean": ndvi_stats.get("NDVI_mean"),
            "valid_pixel_count": ndvi_stats.get("NDVI_count"),
            "mean_cloud_probability": cloud_prob_mean,
            "spacecraft": image.get("SPACECRAFT_NAME"),
            "mean_zenith_angle": image.get("MEAN_SOLAR_ZENITH_ANGLE"),
            "mean_azimuth_angle": image.get("MEAN_SOLAR_AZIMUTH_ANGLE"),
        }
        return ee.Feature(None, props)
 
    features = collection.map(diagnose)
    result = features.getInfo()
    records = [f["properties"] for f in result["features"]]
 
    log.info("Diagnostics for %d scenes:", len(records))
    for r in records:
        log.info("  %s", r)
    return records

