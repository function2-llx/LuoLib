from torch.nn import functional as torch_f

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.networks import one_hot
from monai.utils import MetricReduction
from umei import SegModel
from umei.datasets.btcv import BTCVArgs
from umei.utils import DataKey

class BTCVModel(SegModel):
    def __init__(self, args: BTCVArgs):
        super().__init__(args)
        self.seg_loss_fn = DiceCELoss(
            include_background=self.args.include_background,
            to_onehot_y=True,
            softmax=True,
            squared_pred=self.args.squared_dice,
            smooth_nr=self.args.dice_nr,
            smooth_dr=self.args.dice_dr,
        )

        # metrics for test
        # self.dice_pre = DiceMetric(include_background=False)
        # self.dice_post = DiceMetric(include_background=False)
        # # self.sd_pre = SurfaceDistanceMetric(include_background=False, symmetric=True)
        # self.sd_post = SurfaceDistanceMetric(include_background=False, symmetric=True)
        # # self.hd95_pre = HausdorffDistanceMetric(include_background=False, percentile=95, directed=False)
        # self.hd95_post = HausdorffDistanceMetric(include_background=False, percentile=95, directed=False)
        # # self.resampler = monai.transforms.SpatialResample()
        self.metrics = {
            'dice': DiceMetric(include_background=False),
        }
        self.results = {}

    def on_test_epoch_start(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
        self.results = {}
        # # self.dice_pre.reset()
        # self.dice_post.reset()
        # # self.sd_pre.reset()
        # self.sd_post.reset()
        # # self.hd95_pre.reset()
        # self.hd95_post.reset()

    def test_step(self, batch, batch_idx, *args, **kwargs):
        seg = batch[DataKey.SEG]
        seg_oh = one_hot(seg, self.args.num_seg_classes)
        pred_logit = self.sw_infer(batch[DataKey.IMG])

        # pred = pred_logit.argmax(dim=1, keepdim=True).to(torch.uint8)
        # from swin_unetr.BTCV.utils import resample_3d
        # pred = resample_3d(pred[0, 0].cpu().numpy(), seg.shape[2:])
        # pred = torch.from_numpy(pred)[None].to(seg.device)
        # # pred = torch_f.interpolate(pred, seg.shape[2:], mode='nearest')
        # pred = self.post_transform(pred)
        # # add dummy batch dim
        # pred_oh = one_hot(pred.view(1, *pred.shape), self.args.num_seg_classes)
        # print('argmax-interpolate', end=' ')
        # for metric in [self.dice_pre, self.sd_pre, self.hd95_pre]:
        #     print(metric(pred_oh, seg_oh).nanmean().item(), end='\n' if metric is self.hd95_pre else ' ')
        pred_logit = torch_f.interpolate(pred_logit, seg.shape[2:], mode='trilinear')
        pred = pred_logit.argmax(dim=1, keepdim=True)
        # pred = self.post_transform(pred[0])
        pred_oh = one_hot(pred, self.args.num_seg_classes)
        for k, metric in self.metrics.items():
            m = metric(pred_oh, seg_oh)
            print(m.nanmean().item() * 100)

    def test_epoch_end(self, *args):
        for k, metric in self.metrics.items():
            m = metric.aggregate(reduction=MetricReduction.MEAN_BATCH)
            m = m.nanmean() * 100
            self.log(f'test/{k}/avg', m, sync_dist=True)
            self.results[k] = m.item()

        # for phase, dice_metric, sd_metric, hd95_metric in [
        #     # ('pre', self.dice_pre, self.hd95_pre, self.sd_pre),
        #     ('post', self.dice_post, self.sd_post, self.hd95_post),
        # ]:
        #     dice = dice_metric.aggregate(reduction=MetricReduction.MEAN_BATCH) * 100
        #     sd = sd_metric.aggregate(reduction=MetricReduction.MEAN_BATCH)
        #     hd95 = hd95_metric.aggregate(reduction=MetricReduction.MEAN_BATCH)
        #     self.log(f'test/dice-{phase}/avg', dice.nanmean(), sync_dist=True)
        #     self.log(f'test/sd-{phase}/avg', sd.nanmean(), sync_dist=True)
        #     self.log(f'test/hd95-{phase}/avg', hd95.nanmean(), sync_dist=True)
