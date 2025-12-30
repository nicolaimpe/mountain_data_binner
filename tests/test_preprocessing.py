import numpy as np
import pytest
import rasterio
import xarray as xr
from affine import Affine

from mountain_variability_drivers.preprocessing import preprocess_topography

"""Minimal representative example

DEM:
x0,y0 = 0,7
[[[0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 1. 0. 0. 0.]
  [0. 0. 1. 2. 1. 0. 0.]
  [0. 0. 0. 1. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0.]]]

distributed data:
same resolution, shift of 1 in c and y direction x0,y0 = 1,6

DEM regridded on distributed data transform expected:

[[[0. 0. 0. 0. 0.]
  [0. 0. 1. 0. 0.]
  [0. 1. 2. 1. 0.]
  [0. 0. 1. 0. 0.]
  [0. 0. 0. 0. 0.]]]

Slope map generated from regridded DEM expected [°]:
[[[-9999.      -9999.      -9999.      -9999.      -9999.     ]
  [-9999.         35.26439    45.         35.26439 -9999.     ]
  [-9999.         45.          0.         45.      -9999.     ]
  [-9999.         35.26439    45.         35.26439 -9999.     ]
  [-9999.      -9999.      -9999.      -9999.      -9999.     ]]]

Aspect map generated from regridded DEM expected [°]
[[[-9999. -9999. -9999. -9999. -9999.]
  [-9999.   315.     0.    45. -9999.]
  [-9999.   270. -9999.    90. -9999.]
  [-9999.   225.   180.   135. -9999.]
  [-9999. -9999. -9999. -9999. -9999.]]]
"""


@pytest.fixture(scope="session")
def test_dem_file(tmp_path_factory):
    dem_data = np.pad(np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]]), pad_width=2)
    file_name = tmp_path_factory.mktemp("data") / "dem.tif"
    transform = Affine(1, 0, 0, 0, -1, 7)
    with rasterio.open(
        file_name, "w", width=7, height=7, count=1, dtype=np.float32, nodata=-9999, transform=transform, crs="EPSG:4326"
    ) as dst:
        dst.write(dem_data, 1)
    return file_name


@pytest.fixture(scope="session")
def test_dem_file_regrid_true(tmp_path_factory):
    dem_regrid_data = np.pad(np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]]), pad_width=1)
    file_name = tmp_path_factory.mktemp("data") / "dem_regrid_true.tif"
    transform = Affine(1, 0, 1, 0, -1, 6)
    with rasterio.open(
        file_name, "w", width=5, height=5, count=1, dtype=np.float32, nodata=-9999, transform=transform, crs="EPSG:4326"
    ) as dst:
        dst.write(dem_regrid_data, 1)
    return file_name


@pytest.fixture(scope="session")
def test_slope_file_true(tmp_path_factory):
    slope_data = np.pad(
        np.array([[35.26439, 45, 35.26439], [45, 0, 45], [35.26439, 45, 35.26439]]), pad_width=1, constant_values=-9999
    )

    file_name = tmp_path_factory.mktemp("data") / "slope_true.tif"
    transform = Affine(1, 0, 1, 0, -1, 6)
    with rasterio.open(
        file_name, "w", width=5, height=5, count=1, dtype=np.float32, nodata=-9999, transform=transform, crs="EPSG:4326"
    ) as dst:
        dst.write(slope_data, 1)
    return file_name


@pytest.fixture(scope="session")
def test_aspect_file_true(tmp_path_factory):
    aspect_data = np.pad(np.array([[315, 0, 45], [270, -9999, 90], [225, 180, 135]]), pad_width=1, constant_values=-9999)

    file_name = tmp_path_factory.mktemp("data") / "aspect_true.tif"
    transform = Affine(1, 0, 1, 0, -1, 6)
    with rasterio.open(
        file_name, "w", width=5, height=5, count=1, dtype=np.float32, nodata=-9999, transform=transform, crs="EPSG:4326"
    ) as dst:
        dst.write(aspect_data, 1)
    return file_name


@pytest.fixture(scope="session")
def test_distributed_data_file(tmp_path_factory):
    distributed_data = np.ones(shape=(5, 5))
    file_name = tmp_path_factory.mktemp("data") / "distributed.tif"
    transform = Affine(1, 0, 1, 0, -1, 6)
    with rasterio.open(
        file_name, "w", width=5, height=5, count=1, dtype=np.float32, nodata=-9999, transform=transform, crs="EPSG:4326"
    ) as dst:
        dst.write(distributed_data, 1)
    return file_name


def test_preprocess_topography(
    test_dem_file,
    test_dem_file_regrid_true,
    test_slope_file_true,
    test_aspect_file_true,
    test_distributed_data_file,
    tmp_path_factory,
):
    regrid_dem, regrid_slope, regrid_aspect = preprocess_topography(
        input_dem_filepath=test_dem_file,
        distributed_data_filepath=test_distributed_data_file,
        output_folder=tmp_path_factory.mktemp("data"),
    )

    assert np.array_equal(
        xr.open_dataarray(regrid_dem, engine="rasterio").values, rasterio.open(test_dem_file_regrid_true).read()
    )

    assert np.array_equal(
        xr.open_dataarray(regrid_slope, engine="rasterio", mask_and_scale=False).values,
        rasterio.open(test_slope_file_true).read(),
    )
    assert np.array_equal(
        xr.open_dataarray(regrid_aspect, engine="rasterio", mask_and_scale=False).values,
        rasterio.open(test_aspect_file_true).read(),
    )
