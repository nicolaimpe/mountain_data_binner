from dataclasses import dataclass

import numpy as np

from mountain_data_binner.mountain_binner import MountainBinner, MountainBinnerConfig
from mountain_data_binner.preprocessing import preprocess_topography


class SemidistributedError(Exception):
    pass


@dataclass
class SemidistributedConfig:
    slope_map_path: str | None = None
    aspect_map_path: str | None = None
    dem_path: str | None = None
    regular_8_aspects: bool = True


class Semidistributed(MountainBinner):
    def __init__(self, config: SemidistributedConfig):
        super().__init__(
            MountainBinnerConfig(
                slope_map_path=config.slope_map_path,
                aspect_map_path=config.aspect_map_path,
                dem_path=config.dem_path,
                regular_8_aspects=config.regular_8_aspects,
                forest_mask_path=None,
            )
        )

    @classmethod
    def from_dem_filepath(cls, dem_filepath: str, distributed_data_filepath: str, output_folder: str):
        output_dem_filepath, output_slope_filepath, output_aspect_filepath = preprocess_topography(
            input_dem_filepath=dem_filepath, distributed_data_filepath=distributed_data_filepath, output_folder=output_folder
        )
        return cls(
            SemidistributedConfig(
                slope_map_path=output_slope_filepath,
                aspect_map_path=output_aspect_filepath,
                dem_path=output_dem_filepath,
                regular_8_aspects=True,
            )
        )

    @staticmethod
    def create_default_bin_dict(altitude_step: int = 300, altitude_max: int = 4801):
        return dict(
            slope=MountainBinner.default_slope_bands(),
            aspect=MountainBinner.regular_8_aspect_bins(),
            altitude=MountainBinner.altitude_bands(altitude_step=altitude_step, altitude_max=altitude_max),
        )

    @staticmethod
    def create_user_bin_dict(slope_edges: np.ndarray, aspect_edges: np.ndarray, altitude_edges: np.ndarray):
        if np.any(altitude_edges < 0) or np.any(slope_edges < 0):
            raise SemidistributedError(
                f"Negative altitudes and slopes not supported. Your altitude {altitude_edges}. Your slopes {slope_edges}"
            )
        return dict(
            slope=MountainBinner.user_bins(bin_edges=slope_edges),
            aspect=MountainBinner.user_bins(bin_edges=aspect_edges),
            altitude=MountainBinner.user_bins(bin_edges=altitude_edges),
        )
