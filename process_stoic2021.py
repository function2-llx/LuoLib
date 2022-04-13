from copy import deepcopy
import json
from pathlib import Path
from typing import Dict

import SimpleITK as sitk
from evalutils.evalutils import Algorithm
from evalutils.validators import UniqueImagesValidator, UniquePathIndicesValidator
import lungmask.mask as lungmask
import numpy as np
import pandas as pd
from ruamel.yaml import YAML
import pytorch_lightning as pl

from umei.datasets import Stoic2021Args, Stoic2021Model
from umei.datasets.stoic2021.datamodule import Stoic2021DataModule
from umei.model import build_encoder
from umei.utils import UMeIParser

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
        args.dataloader_num_workers = 0
        self.args = args
        self.lungmask_model = lungmask.get_model('unet', 'R231').eval()

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

    def process_case(self, *, idx: int, case: pd.DataFrame) -> Dict:
        img_path = Path(case['path'])
        self.lungmask(img_path)
        age, sex = self.collect_clinical(img_path)
        data = {
            self.args.img_key: img_path,
            self.args.mask_key: MASK_PATH,
            self.args.clinical_key: np.array([age / 100, sex, sex ^ 1]),
        }

        def get_dataloaders():
            ret = [Stoic2021DataModule(self.args, predict_case=data).predict_dataloader()]
            args = deepcopy(self.args)
            # keep original shape
            args.sample_size = args.sample_slices = -1
            ret.append(Stoic2021DataModule(args, predict_case=data).predict_dataloader())
            return ret

        dataloaders = get_dataloaders()
        trainer = pl.Trainer(
            logger=False,
            gpus=1,
            benchmark=True,
        )
        pred = {
            'severity_pred': 0,
            'positivity_pred': 0,
        }
        encoder = build_encoder(self.args)
        num_models = 0
        for model_ckpt in Path(self.args.output_dir).glob('fold*/run0/*.ckpt'):
            model = Stoic2021Model.load_from_checkpoint(str(model_ckpt), args=self.args, encoder=encoder)
            num_models += len(dataloaders)
            for result in trainer.predict(model, dataloaders):
                for k, v in result[0].items():
                    pred[k] += v.item()

        return {
            SEVERE_OUTPUT_NAME: pred['severity_pred'] / num_models,
            COVID_OUTPUT_NAME: pred['positivity_pred'] / num_models,
        }

    def predict(self, *args, **kwargs):
        pass

    def save(self):
        if len(self._case_results) > 1:
            raise RuntimeError("Multiple case prediction not supported with single-value output interfaces.")
        case_result = self._case_results[0]

        for output_file, result in case_result.items():
            with open(str(self._output_path / output_file) + '.json', "w") as f:
                json.dump(result, f)

def main():
    import sys
    sys.argv.insert(1, 'conf/stoic2021/infer.yml')
    parser = UMeIParser((Stoic2021Args, ), use_conf=True)
    args: Stoic2021Args = parser.parse_args_into_dataclasses()[0]
    StoicAlgorithm(args).process()

if __name__ == "__main__":
    main()
