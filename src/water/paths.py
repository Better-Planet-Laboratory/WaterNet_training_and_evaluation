from pathlib import Path
import os
import sys
import yaml

def open_yaml(file_name: Path):
    with open(f'{file_name}', 'r') as file:
        obj = yaml.safe_load(file)
    return obj

package_path = Path(__file__).parent
src_path = package_path.parent
# base_path = bridges_package_path.parent
if (src_path.parent / 'configuration_files').exists():
    base_path = src_path.parent
elif (Path(sys.path[0]) / 'configuration_files').exists():
    base_path = Path(sys.path[0])
elif (Path(sys.path[0]).parent / 'configuration_files').exists():
    base_path = Path(sys.path[0]).parent
elif (Path(os.getcwd()) / 'configuration_files').exists():
    base_path = Path(os.getcwd())
elif (Path(os.getcwd()).parent / 'configuration_files').exists():
    base_path = Path(os.getcwd()).parent
elif (Path(os.getcwd()).parent.parent / 'configuration_files').exists():
    base_path = Path(os.getcwd()).parent.parent
else:
    raise Exception(
        'Can\'t find data directory... '
        'A directory named "configuration_files" must exist'
        'in the same directory as this script '
        'or in this script\'s parent directory'
        f'Checked:'
        f'{os.getcwd()}'
        f'{sys.path[0]}'
    )


class Proj_paths:
    """
    paths for the project
    """
    def __init__(self,
                 base: Path = base_path
                 ) -> None:
        self.base_path = base
        self.configuration_files = self.base_path / 'configuration_files'
        self._path_config = {}
        if (self.configuration_files / 'path_configuration.yaml').exists():
            self._path_config = open_yaml(self.configuration_files / 'path_configuration.yaml')
            if self._path_config is None:
                self._path_config = {}
        self.data = base / 'data'
        self.training_data = self.add_directory('training_data', self.data)
        self.world_info = self.add_directory('world_info', self.data)
        self.deploy_data = self.add_directory('deploy_data', self.training_data)
        self.elevation_data = self.add_directory('elevation', self.deploy_data)
        self.sentinel_unmerged = self.add_directory('sentinel_unmerged', self.deploy_data)
        self.sentinel_merged = self.add_directory('sentinel_merged', self.deploy_data)
        self.tdx_basins = self.add_directory('tdx_basins', self.deploy_data)
        self.tdx_streams = self.add_directory('tdx_streams', self.deploy_data)
        self.hu4_data = self.add_directory('hu4_data', self.training_data)
        self.hu4_parquet = self.add_directory('hu4_parquet', self.training_data)
        self.hu4_hull = self.add_directory('hu4_hull', self.training_data)
        self.hu4_hulls_parquet = self.add_file('hu4_hulls.parquet', self.training_data)
        self.sentinel_tiles_parquet = self.add_file('sentinel_tiles.parquet', self.world_info)
        self.world_boundaries_parquet = self.add_file('world_boundaries.parquet', self.world_info)

        self.elevation_cut = self.add_directory('elevation_cut', self.training_data)
        self.sentinel_cut = self.add_directory('sentinel_cut', self.training_data)
        self.waterways_burned = self.add_directory('waterways_burned', self.training_data)
        self.model_inputs_832 = self.add_directory('model_inputs_832', self.training_data)
        self.model_inputs_224 = self.add_directory('model_inputs_224', self.training_data)

    def add_directory(self, directory_name: str, directory_parent: Path):
        if directory_name not in self._path_config:
            directory_path = directory_parent / directory_name
        else:
            directory_path = Path(self._path_config[directory_name])
        if not directory_path.exists() and not directory_path.is_symlink():
            directory_path.mkdir()
        return directory_path

    def add_file(self, file_name: str, directory_path: Path):
        if file_name not in self._path_config:
            file_path = directory_path / file_name
        else:
            file_path = Path(self._path_config[file_name])
        return file_path


ppaths = Proj_paths()