from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import xarray as xr
from xarray.groupers import BinGrouper, UniqueGrouper

from mountain_data_binner.preprocessing import preprocess


class MountainBinnerError(Exception):
    pass


@dataclass
class MountainBinnerConfig:
    slope_map_path: str | None = None
    aspect_map_path: str | None = None
    dem_path: str | None = None
    forest_mask_path: str | None = None
    regular_8_aspects: bool = True


class MountainBinner:
    def __init__(self, config: MountainBinnerConfig):
        self.config = config

    @staticmethod
    def aspect_map_transform(aspect_map: xr.DataArray) -> xr.DataArray:
        """
        Aspect map in degrees azimuth

        Transform the aspect map so that its values are monotonically incresing from N to NW,
        without dividing the North in two bins (NNW [337.5-360] and NNE [0-315])
        This is convenient for BinGrouper object

        """
        # Transform the aspect map so that its values are monotonically incresing from N to NW,
        # without dividing the North in two bins (NNW [337.5-360] and NNE [0-315])
        # This is convenient for BinGrouper object

        aspect_map = aspect_map.where(aspect_map < 360 - 22.5, aspect_map - 360)
        return aspect_map

    @staticmethod
    def user_bins(bin_edges: np.ndarray) -> BinGrouper:
        return BinGrouper(
            bins=bin_edges,
            labels=MountainBinner.create_labels_from_bin_edges(bin_edges),
            include_lowest=True,
            right=False,
        )

    @staticmethod
    def create_labels_from_bin_edges(bin_edges: np.ndarray) -> List[str]:
        bin_labels = []
        for i in range(len(bin_edges) - 1):
            bin_labels.append(f"{bin_edges[i]} - {bin_edges[i + 1]}")
        return bin_labels

    @staticmethod
    def slope_bands(slope_sampling_step: int = 20) -> BinGrouper:
        bin_edges = np.arange(0, 90, slope_sampling_step)
        return BinGrouper(
            bin_edges, labels=MountainBinner.create_labels_from_bin_edges(bin_edges), include_lowest=True, right=False
        )

    @staticmethod
    def default_slope_bands() -> BinGrouper:
        bin_edges = np.array([0, 10, 30, 50])
        return BinGrouper(
            bin_edges, labels=MountainBinner.create_labels_from_bin_edges(bin_edges), include_lowest=True, right=False
        )

    @staticmethod
    def forest_mask_bins() -> BinGrouper:
        return BinGrouper([0, 1, 2], labels=["open", "forest"], include_lowest=True, right=False)

    @staticmethod
    def altitude_bands(altitude_step: int = 300, altitude_min: int = 0, altitude_max: int = 4800) -> BinGrouper:
        bin_edges = np.arange(altitude_min, altitude_max, altitude_step)

        return BinGrouper(
            bin_edges, labels=MountainBinner.create_labels_from_bin_edges(bin_edges), include_lowest=True, right=False
        )

    def bins_max(self, MountainBinner_data: xr.DataArray, coordinate_name: str) -> np.ndarray:
        return np.array(
            [float(bin_max_str.split("-")[1]) for bin_max_str in MountainBinner_data.coords[coordinate_name].values]
        )

    def bins_min(self, MountainBinner_data: xr.DataArray, coordinate_name: str) -> np.ndarray:
        # print(MountainBinner_data.sel(altitude_bins=1))
        return np.array(
            [float(bin_min_str.split("-")[0]) for bin_min_str in MountainBinner_data.coords[coordinate_name].values]
        )

    @staticmethod
    def regular_8_aspect_bins() -> BinGrouper:
        return BinGrouper(
            np.arange(-45 / 2, 360, 45),
            labels=np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
        )

    @staticmethod
    def create_default_bin_dict(altitude_step: int = 300, altitude_max: int = 4801):
        return dict(
            slope=MountainBinner.default_slope_bands(),
            aspect=MountainBinner.regular_8_aspect_bins(),
            altitude=MountainBinner.altitude_bands(altitude_step=altitude_step, altitude_max=altitude_max),
            forest_mask=MountainBinner.forest_mask_bins(),
        )

    @staticmethod
    def create_user_bin_dict(
        slope_edges: np.ndarray, aspect_edges: np.ndarray, altitude_edges: np.ndarray, landcover_classes: np.ndarray
    ):
        if np.any(altitude_edges < 0) or np.any(slope_edges < 0):
            raise MountainBinnerError(
                f"Negative altitudes and slopes not supported. Your altitude {altitude_edges}. Your slopes {slope_edges}"
            )
        landcover_classes_sorted = np.sort(landcover_classes)
        return dict(
            slope=MountainBinner.user_bins(bin_edges=slope_edges),
            aspect=MountainBinner.user_bins(bin_edges=aspect_edges),
            altitude=MountainBinner.user_bins(bin_edges=altitude_edges),
            # landcover classes are defined on a bin whose lowest values correspond to the discrete value of the class
            # we need to define an extra bin edge for the last class in order to keep the bin definition
            # this way is should be transparent to users
            forest_mask=BinGrouper(
                bins=np.array([*landcover_classes_sorted, landcover_classes_sorted[-1] + 1]),
                labels=landcover_classes_sorted,
                include_lowest=True,
                right=False,
            ),
        )

    @classmethod
    def from_dem_and_forest_mask_filepath(
        cls, dem_filepath: str, forest_mask_filepath: str, distributed_data_filepath: str, output_folder: str
    ):
        output_dem_filepath, output_slope_filepath, output_aspect_filepath, output_forest_mask_filepath = preprocess(
            input_dem_filepath=dem_filepath,
            forest_mask_filepath=forest_mask_filepath,
            distributed_data_filepath=distributed_data_filepath,
            output_folder=output_folder,
        )
        return cls(
            MountainBinnerConfig(
                slope_map_path=output_slope_filepath,
                aspect_map_path=output_aspect_filepath,
                dem_path=output_dem_filepath,
                regular_8_aspects=True,
                forest_mask_path=output_forest_mask_filepath,
            )
        )

    def stack_auxiliary_data(self, distributed_data: xr.DataArray | xr.Dataset):
        if self.config.slope_map_path is not None:
            slope_map = xr.open_dataarray(self.config.slope_map_path)
            dataset = distributed_data.assign(slope=slope_map)

        if self.config.aspect_map_path is not None:
            aspect_map = xr.open_dataarray(self.config.aspect_map_path)
            if self.config.regular_8_aspects:
                aspect_map = MountainBinner.aspect_map_transform(aspect_map)
            dataset = dataset.assign(aspect=aspect_map)

        if self.config.dem_path is not None:
            dem_map = xr.open_dataarray(self.config.dem_path)
            dataset = dataset.assign(altitude=dem_map)

        if self.config.forest_mask_path is not None:
            forest_mask = xr.open_dataarray(self.config.forest_mask_path)
            dataset = dataset.assign(forest_mask=forest_mask)

        # Drop band dimension if rioxarray was used as engine
        if "band" in dataset.dims:
            dataset = dataset.sel(band=1).drop_vars("band")

        return dataset

    def prepare(self, distributed_data: xr.DataArray | xr.Dataset, analysis_bin_dict: Dict[str, BinGrouper]) -> xr.Dataset:
        variable_and_auxiliary = self.stack_auxiliary_data(distributed_data=distributed_data)
        return variable_and_auxiliary.groupby(analysis_bin_dict)

    def rename_coords(self, MountainBinner_data: xr.DataArray | xr.Dataset):
        sd = MountainBinner_data
        if self.config.dem_path:
            sd = sd.assign_coords(altitude_min=("altitude_bins", self.bins_min(sd, "altitude_bins")))
            sd = sd.assign_coords(altitude_max=("altitude_bins", self.bins_max(sd, "altitude_bins")))
            sd = sd.set_xindex("altitude_min")
            sd = sd.set_xindex("altitude_max")
        if self.config.slope_map_path:
            sd = sd.assign_coords(slope_min=("slope_bins", self.bins_min(sd, "slope_bins")))
            sd = sd.assign_coords(slope_max=("slope_bins", self.bins_max(sd, "slope_bins")))
            sd = sd.set_xindex("slope_min")
            sd = sd.set_xindex("slope_max")
        if self.config.aspect_map_path:
            if not self.config.regular_8_aspects:
                sd = sd.assign_coords(aspect_min=("aspect_bins", self.bins_min(sd, "aspect_bins")))
                sd = sd.assign_coords(aspect_max=("aspect_bins", self.bins_max(sd, "aspect_bins")))
                sd = sd.set_xindex("aspect_min")
                sd = sd.set_xindex("aspect_max")
            else:
                sd = sd.rename({"aspect_bins": "aspect"})
        if self.config.forest_mask_path:
            sd = sd.rename({"forest_mask_bins": "landcover"})
        return sd

    def transform(
        self,
        distributed_data: xr.DataArray | xr.Dataset,
        analysis_bin_dict: Dict[str, BinGrouper],
        function: Callable,
        *args,
    ):
        transformed = self.prepare(
            distributed_data=distributed_data,
            analysis_bin_dict=analysis_bin_dict,
        ).map(func=function, args=args)
        postprocess = self.rename_coords(transformed)

        return postprocess
