import numpy as np
import pytorch_lightning as pl

import monai
from monai import transforms as monai_t
from umei.datasets.btcv.datamodule_omega import BTCVDataModule
from umei.datasets.btcv.omega import BTCVExpConf
from umei.omega import parse_exp_conf

def main():
    conf = parse_exp_conf(BTCVExpConf)
    pl.seed_everything(conf.seed)
    monai.utils.misc._seed = conf.seed
    datamodule = BTCVDataModule(conf)
    from tqdm import tqdm
    for data in tqdm(datamodule.train_data()):
        from umei.utils.index_tracker import IndexTracker
        from umei.utils import DataKey
        loader = monai_t.Compose(
            datamodule.loader_transform()
            + datamodule.normalize_transform(),
        )
        data = loader(data)
        # img = data[DataKey.IMG][0]
        # seg = data[DataKey.SEG][0]
        # IndexTracker(img, seg)
        from copy import deepcopy

        aug = monai_t.Compose(datamodule.aug_transform())
        aug_data = aug(deepcopy(data))
        img = aug_data[DataKey.IMG][0]
        seg = aug_data[DataKey.SEG][0]
        meta = img.meta
        center = meta['crop center']
        rotate_params = meta['rotate']
        scale_params = meta['scale']
        IndexTracker(img, seg, title=f'center: {center}\n'
                                     f'rotate: {rotate_params}\n'
                                     f'scale: {scale_params}')

        # from monai.utils import GridSampleMode
        # from monai.utils import GridSamplePadMode
        # bf = monai_t.Compose([
        #     monai_t.SpatialCropD(
        #         [DataKey.IMG, DataKey.SEG],
        #         center,
        #         conf.sample_shape,
        #     ),
        #     monai_t.AffineD(
        #         [DataKey.IMG, DataKey.SEG],
        #         [0, 0, *rotate_params],
        #         scale_params=[*scale_params, 1],
        #         mode=[GridSampleMode.BILINEAR, GridSampleMode.NEAREST],
        #         padding_mode=GridSamplePadMode.ZEROS,
        #     )
        # ])
        # data = bf(data)
        # img = data[DataKey.IMG][0]
        # seg = data[DataKey.SEG][0]
        # IndexTracker(img, seg)

if __name__ == '__main__':
    main()
