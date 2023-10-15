
import logging
import os
import random
from typing import List, Union

from .base_dataset import BaseDataset
from .transforms.transforms_factory import create_transforms, run_transforms

__all__ = ["PGNetDataset"]
_logger = logging.getLogger(__name__)


class PGNetDataset(BaseDataset):
    def __init__(
        self,
        is_train: bool = True,
        data_dir: Union[str, List[str]] = None,
        label_file: Union[List, str] = None,
        sample_ratio: Union[List, float] = 1.0,
        shuffle: bool = None,
        transform_pipeline: List[dict] = None,
        output_columns: List[str] = None,
        **kwargs,
    ):
        super().__init__(data_dir=data_dir, label_file=label_file, output_columns=output_columns)

        # check args
        if isinstance(sample_ratio, float):
            sample_ratio = [sample_ratio] * len(self.label_file)

        shuffle = shuffle if shuffle is not None else is_train

        # load date file list
        self.data_list = self.load_data_list(self.label_file, sample_ratio, shuffle)

        # create transform
        if transform_pipeline is not None:
            global_config = dict(is_train=is_train)
            self.transforms = create_transforms(transform_pipeline, global_config)
        else:
            raise ValueError("No transform pipeline is specified!")

        # prefetch the data keys, to fit GeneratorDataset
        _data = self.data_list[0].copy()  # WARNING: shallow copy. Do deep copy if necessary.
        _data = run_transforms(_data, transforms=self.transforms)
        _available_keys = list(_data.keys())

        if output_columns is None:
            self.output_columns = _available_keys
        else:
            self.output_columns = []
            for k in output_columns:
                if k in _data:
                    self.output_columns.append(k)
                else:
                    raise ValueError(
                        f"Key '{k}' does not exist in data (available keys: {_data.keys()}). "
                        "Please check the name or the completeness transformation pipeline."
                    )

        final_data_list = []
        for index in range(len(self.data_list)):
            data = self.data_list[index].copy()
            try:
                data = run_transforms(data, transforms=self.transforms)
            except Exception as e:
                _logger.warning(f"Error occurred while processing the image: {self.data_list[index]['img_path']}\n {e}")
            else:
                final_data_list.append(data)
        self.data_list = final_data_list

    def __getitem__(self, index):
        data = self.data_list[index]
        output_tuple = tuple(data[k] for k in self.output_columns)
        return output_tuple

    def load_data_list(
        self, label_file: List[str], sample_ratio: List[float], shuffle: bool = False, **kwargs
    ) -> List[dict]:
        """Load data list from label_file which contains infomation of image paths and annotations
        Args:
            label_file: annotation file path(s)
            sample_ratio sample ratio for data items in each annotation file
            shuffle: shuffle the data list
        Returns:
            data (List[dict]): A list of annotation dict, which contains keys: img_path, annot...
        """

        # parse image file path and annotation and load
        data_list = []
        for idx, label_fp in enumerate(label_file):
            img_dir = self.data_dir[idx]
            with open(label_fp, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if shuffle:
                    lines = random.sample(lines, round(len(lines) * sample_ratio[idx]))
                else:
                    lines = lines[: round(len(lines) * sample_ratio[idx])]

                for line in lines:
                    img_name, annot_str = self._parse_annotation(line)

                    img_path = os.path.join(img_dir, img_name)
                    assert os.path.exists(img_path), "{} does not exist!".format(img_path)

                    data = {"img_path": img_path, "label": annot_str}
                    data_list.append(data)

        return data_list

    def _parse_annotation(self, data_line: str):
        data_line_tmp = data_line.strip()
        if "\t" in data_line_tmp:
            img_name, annot_str = data_line.strip().split("\t")
        elif " " in data_line_tmp:
            img_name, annot_str = data_line.strip().split(" ")
        else:
            raise ValueError(
                "Incorrect label file format: the file name and the label should be separated by " "a space or tab"
            )

        return img_name, annot_str
