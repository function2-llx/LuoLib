from pathlib import Path

from toolz import itertoolz as itz
from torch.nn import functional as torch_f

from monai import transforms as monai_t
from monai.data import MetaTensor
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.networks import one_hot
from monai.utils import ImageMetaKey, MetricReduction, TraceKeys

from umei.models import SegModel
from umei.utils import DataKey
from .omega import BTCVExpConf

class BTCVModel(SegModel):
    conf: BTCVExpConf

    def __init__(self, conf: BTCVExpConf):
        super().__init__(conf)
        self.test_metrics = {
            'dice': DiceMetric(include_background=True),
            'sd': SurfaceDistanceMetric(include_background=True, symmetric=True, no_inf=True),
            'hd95': HausdorffDistanceMetric(include_background=True, percentile=95, directed=False),
        }
        self.results = {}
        self.case_results = {
            k: []
            for k in self.test_metrics.keys()
        }

    def on_test_epoch_start(self) -> None:
        for k in self.test_metrics.keys():
            self.test_metrics[k].reset()
            self.case_results[k].clear()
        self.results = {}

    def test_step(self, batch, batch_idx, *args, **kwargs):
        img: MetaTensor = batch[DataKey.IMG]
        seg: MetaTensor = batch[DataKey.SEG]
        if img.shape[0] > 1:
            raise NotImplementedError
        pred_value = self.infer(img)
        to_pad = None
        for op in reversed(img.applied_operations[0]):
            match op[TraceKeys.CLASS_NAME]:
                case 'SpatialPad':
                    padded = op[TraceKeys.EXTRA_INFO]["padded"]
                    print('\npadded:', padded)
                    roi_start = [i[0] for i in padded[1:]]
                    roi_end = [i - j[1] for i, j in zip(pred_value.shape[2:], padded[1:])]
                    cropper = monai_t.SpatialCrop(roi_start=roi_start, roi_end=roi_end)
                    with cropper.trace_transform(False):
                        pred_value[0] = cropper(pred_value[0])
                case 'SpatialResample':
                    pred_value = torch_f.interpolate(pred_value, op[TraceKeys.ORIG_SIZE], mode='trilinear')
                case 'CropForeground':
                    to_pad = list(itz.partition(2, op[TraceKeys.EXTRA_INFO]['cropped']))

        pred = pred_value.argmax(dim=1, keepdim=True).int()
        # pred = self.post_transform(pred[0])
        pred = torch_f.pad(pred, list(itz.concat(reversed(to_pad))))
        seg_oh = one_hot(seg, self.conf.num_seg_classes)
        pred_oh = one_hot(pred, self.conf.num_seg_classes)
        for k, metric in self.test_metrics.items():
            m = metric(pred_oh, seg_oh)
            for i in range(m.shape[0]):
                self.case_results[k].append('\t'.join(map(str, m[i].tolist())))
            avg = m[:, 1:].nanmean().item()
            if k == 'dice':
                avg *= 100
            print(k, avg)
        if self.conf.export_seg_pred:
            import nibabel as nib
            pred_np = pred.cpu().numpy()
            affine_np = seg.affine.numpy()
            for i in range(pred.shape[0]):
                img_path = Path(img.meta[ImageMetaKey.FILENAME_OR_OBJ][i])
                case = img_path.with_suffix('').stem[3:]
                nib.save(
                    nib.Nifti1Image(pred_np[i, 0], affine_np[i]),
                    Path(self.trainer.log_dir) / f'seg{case}.nii.gz',
                )

    def test_epoch_end(self, *_):
        conf = self.conf

        for k, metric in self.test_metrics.items():
            m = metric.aggregate(reduction=MetricReduction.MEAN_BATCH)
            m = m[1:].nanmean()
            if k == 'dice':
                m *= 100
            self.log(f'test/{k}/avg', m, sync_dist=True)
            self.results[k] = m.item()

        with open(conf.log_dir / 'results.txt', 'w') as f:
            print('\t'.join(self.results.keys()), file=f)
            print('\t'.join(map(str, self.results.values())), file=f)
        with open(conf.log_dir / 'case-results.txt', 'w') as f:
            for k, v in self.case_results.items():
                print(k, file=f)
                print(*v, sep='\n', file=f)
