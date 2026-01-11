# mountain_data_binner
Efficient xarray-based workflow to perform data binning of a geospatial quantity in a mountain landscape.

## Description
Map distributed (gridded) information into a multi-dimensional space whose coordinates are the main drivers of
spatial  variability in a mountain landscape: topography (elevation, aspect, slope), landcover (presence of canopy).

```bash
$ tree
├── data
├── notebooks
│   └── example_usage.ipynb         : Use cases
├── scripts
│   └── process_mnt.sh              : GDAL based script to regrid and prepare data (same of preprocessing.py)
├── src
│   └── mountain_data_binner
│       ├── __init__.py
│       ├── mountain_binner.py      : class to project a geospatial dataset into the binned space
│       ├── preprocessing.py        : regrid and prepare data for efficient binning
│       └── semidistributed.py      : particular case of mountain_binner where we are interested in a topographic 
|                                     representation (semidistributed geometry)
└──  tests
```
## Installation

```bash
git clone git@github.com:nicolaimpe/mountain_data_binner.git
cd mountain_data_binner
pip install .
```

## Usage
```python
import xarray as xr
from mountain_data_binner.mountain_binner import MountainBinner, MountainBinnerConfig
import os

input_dem_filepath = "../data/DEM_ALPES_NORD_L93_250m_bilinear.tif"
forest_mask_filepath = "../data/FOREST_MASK_ALPES_NORD_EPSG3035_100m.tif"
output_folder = "../output_folder"
snow_cover_data_filepath = "../data/EOFR63JPSS_multisat_20251122.nc"

os.makedirs(output_folder, exist_ok=True)


def target_fun(data: xr.Dataset) -> xr.DataArray:
    """Reduction function, a simple mean."""
    return data.data_vars["snow_cover_fraction"].mean()

# When init it preprocess DEM and forest data using GDAL
mountain_binner = MountainBinner.from_dem_and_forest_mask_filepath(
    dem_filepath=input_dem_filepath,
    distributed_data_filepath=snow_cover_data_filepath,
    output_folder=output_folder,
    forest_mask_filepath=forest_mask_filepath,
)
s2m_dict = MountainBinner.create_default_bin_dict(altitude_max=4801, altitude_step=300)
snow_cover_dataset = xr.open_dataset(snow_cover_data_filepath, engine="rasterio")
# rescale snow cover fraction to 0 - 100%
snow_cover_dataset = snow_cover_dataset.where(snow_cover_dataset.data_vars["snow_cover_fraction"].values <= 200) / 2

# Binning operation: mapping in topographic and landcover coordinates
binned_dataset = mountain_binner.transform(distributed_data=snow_cover_dataset, bin_dict=s2m_dict, function=target_fun)
print(binned_dataset)
```

```output
<xarray.DataArray 'snow_cover_fraction' (slope_bins: 3, aspect: 8,
                                         altitude_bins: 16, landcover: 2)> Size: 6kB
array([[[[5.69264054e-01, 0.00000000e+00],
         [1.06905737e+01, 6.84905672e+00],
         [5.61095886e+01, 5.05846138e+01],
         [7.08136368e+01, 7.41290359e+01],
         [7.12741928e+01, 7.14242401e+01],
...

         [7.67692337e+01,            nan],
         [4.08571434e+01,            nan],
         [6.66666641e+01,            nan],
         [5.00000000e+01,            nan],
         [           nan,            nan]]]])
Coordinates:
  * slope_bins     (slope_bins) object 24B '0 - 10' '10 - 30' '30 - 50'
  * aspect         (aspect) object 64B 'N' 'NE' 'E' 'SE' 'S' 'SW' 'W' 'NW'
  * altitude_bins  (altitude_bins) object 128B '0 - 300' ... '4500 - 4800'
  * landcover      (landcover) object 16B 'open' 'forest'
    spatial_ref    int64 8B 0
  * altitude_min   (altitude_bins) float64 128B 0.0 300.0 ... 4.2e+03 4.5e+03
  * altitude_max   (altitude_bins) float64 128B 300.0 600.0 ... 4.5e+03 4.8e+03
  * slope_min      (slope_bins) float64 24B 0.0 10.0 30.0
  * slope_max      (slope_bins) float64 24B 10.0 30.0 50.0
```
See `notebooks/example_usage.ipynb` for use cases.

## Contributing

Contributions are welcome.

PDM is recommended for environment management.

```bash
pip install pdm
pdm install
```

To add a package to the project

```bash
pip add <your_package>
```