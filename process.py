import json
from pathlib import Path
from typing import Callable, Dict

import SimpleITK as sitk
from evalutils.evalutils import Algorithm
from evalutils.validators import UniqueImagesValidator, UniquePathIndicesValidator
import lungmask.mask as lungmask
import numpy as np
import pandas as pd
from ruamel.yaml import YAML
import torch

import monai
from monai.utils import InterpolateMode, NumpyPadMode
import umei
from umei.datasets import Stoic2021Args
from umei.datasets.stoic2021.datamodule import DATASET_ROOT
from umei.utils import UMeIParser
from utils import device, to_input_format

yaml = YAML()

COVID_OUTPUT_NAME = Path("probability-covid-19")
SEVERE_OUTPUT_NAME = Path("probability-severe-covid-19")

MASK_PATH = Path('/input/images/umei-lungmask.nii.gz')
# CROPPED_PATH.parent.mkdir(exist_ok=True, parents=True)

MIN_HU = -1024

class StoicAlgorithm(Algorithm):
    def __init__(self, args: Stoic2021Args):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path=Path("/input/images/ct/"),
            output_path=Path("/output/")
        )

        self.args = args
        self.lungmask_model = lungmask.get_model('unet', 'R231').to(device)
        # load model
        # self.model = I3D(nr_outputs=2)
        # self.model = self.model.to(device)
        # self.model.load_state_dict(torch.load('./algorithm/model_covid.pth', map_location=torch.device(device)))
        # self.model = self.model.eval()

    @staticmethod
    def collect_clinical(filepath: Path) -> tuple[int, int]:
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(filepath))  # Give it the mha file as a string
        reader.LoadPrivateTagsOn()  # Make sure it can get all the info
        reader.ReadImageInformation()  # Get just the information from the file
        age = reader.GetMetaData('PatientAge')
        sex = reader.GetMetaData('PatientSex')
        return int(age[:-1]), int(sex == 'M')

    def lungmask(self, filepath: Path):
        img = sitk.ReadImage(str(filepath))
        mask = lungmask.apply(img, model=self.lungmask_model)
        out = sitk.GetImageFromArray(mask)
        out.CopyInformation(img)
        sitk.WriteImage(out, str(MASK_PATH))

    @property
    def preprocess_transform(self) -> Callable:
        stat_path = DATASET_ROOT / 'stat.yml'
        stat = yaml.load(stat_path)

        return monai.transforms.Compose([
            monai.transforms.LoadImageD([self.args.img_key, self.args.mask_key]),
            monai.transforms.AddChannelD([self.args.img_key, self.args.mask_key]),
            monai.transforms.OrientationD([self.args.img_key, self.args.mask_key], axcodes='RAS'),
            monai.transforms.ThresholdIntensityD('img', threshold=MIN_HU, above=True, cval=MIN_HU),
            monai.transforms.LambdaD('img', lambda x: x - MIN_HU),
            monai.transforms.MaskIntensityD('img', mask_key='mask'),
            monai.transforms.LambdaD('img', lambda x: x + MIN_HU),
            monai.transforms.CropForegroundD(['img', 'mask'], source_key='mask'),
            monai.transforms.NormalizeIntensityD(self.args.img_key, subtrahend=stat['mean'], divisor=stat['std']),
            umei.transforms.SpatialSquarePadD([self.args.img_key, self.args.mask_key], mode=NumpyPadMode.EDGE),
        ])

    @property
    def input_transform(self) -> Callable:
        return monai.transforms.Compose([
            monai.transforms.ResizeD(
                [self.args.img_key, self.args.mask_key],
                spatial_size=[self.args.sample_size, self.args.sample_size, self.args.sample_slices],
                mode=[InterpolateMode.AREA, InterpolateMode.NEAREST],
            ),
            monai.transforms.ConcatItemsD([self.args.img_key, self.args.mask_key], name=self.args.img_key),
            monai.transforms.CastToTypeD([self.args.img_key, self.args.clinical_key], dtype=np.float32),
            monai.transforms.SelectItemsD([self.args.img_key, self.args.clinical_key, self.args.cls_key]),
        ])

    def process_case(self, *, idx: int, case: pd.DataFrame) -> Dict:
        img_path = Path(case['path'])
        self.lungmask(img_path)
        age, sex = self.collect_clinical(img_path)
        data = self.preprocess_transform({
            self.args.img_key: img_path,
            self.args.mask_key: MASK_PATH,
            self.args.clinical_key: np.array([age / 100, sex, sex ^ 1]),
        })

    def predict(self, *args, **kwargs):
        pass

    # def predict(self, *, input_image: SimpleITK.Image) -> Dict:
    #     # pre-processing
    #     input_image = preprocess(input_image)
    #     input_image = to_input_format(input_image)
    #
    #     # run model
    #     with torch.no_grad():
    #         output = torch.sigmoid(self.model(input_image))
    #     prob_covid, prob_severe = unpack_single_output(output)
    #
    #     return {
    #         COVID_OUTPUT_NAME: prob_covid,
    #         SEVERE_OUTPUT_NAME: prob_severe
    #     }

    def save(self):
        if len(self._case_results) > 1:
            raise RuntimeError("Multiple case prediction not supported with single-value output interfaces.")
        case_result = self._case_results[0]

        for output_file, result in case_result.items():
            with open(str(self._output_path / output_file) + '.json', "w") as f:
                json.dump(result, f)

def main():
    import sys
    sys.argv.insert(1, 'conf/stoic2021.yml')
    parser = UMeIParser((Stoic2021Args, ), use_conf=True)
    args: Stoic2021Args = parser.parse_args_into_dataclasses()[0]
    StoicAlgorithm(args).process()

if __name__ == "__main__":
    main()
