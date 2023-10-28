import torch

from monai import transforms as mt
from monai.config import KeysCollection, NdarrayOrTensor
from monai.data import get_track_meta
from monai.utils import convert_to_tensor, ensure_tuple_size

__all__ = [
    "RandGaussianSmooth",
    # 'RandGaussianSmoothD',
]

class RandGaussianSmooth(mt.RandomizableTransform):
    def __init__(
        self,
        sigma_x: tuple[float, float] = (0.25, 1.5),
        sigma_y: tuple[float, float] = (0.25, 1.5),
        sigma_z: tuple[float, float] = (0.25, 1.5),
        prob: float = 0.1,
        approx: str = 'erf',
        *,
        prob_per_channel: float = 0.5,
    ) -> None:
        super().__init__(prob)
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        self.approx = approx
        self.prob_per_channel = prob_per_channel

    def randomize(self, img: torch.Tensor):
        super().randomize(img)
        if not self._do_transform:
            return
        self.gaussian_smooth = []
        for i in range(img.shape[0]):
            if self.R.uniform() < self.prob_per_channel:
                x = self.R.uniform(self.sigma_x[0], self.sigma_x[1])
                y = self.R.uniform(self.sigma_y[0], self.sigma_y[1])
                z = self.R.uniform(self.sigma_z[0], self.sigma_z[1])
                self.gaussian_smooth.append(mt.GaussianSmooth(ensure_tuple_size((x, y, z), img.ndim - 1), self.approx))
            else:
                self.gaussian_smooth.append(None)

    def __call__(self, img: NdarrayOrTensor, randomize: bool = True):
        img_t: torch.Tensor = convert_to_tensor(img, track_meta=get_track_meta())
        if randomize:
            self.randomize(img_t)
        if not self._do_transform:
            return img_t

        for i, gaussian_smooth in enumerate(self.gaussian_smooth):
            if gaussian_smooth is not None:
                img_t[i:i + 1] = gaussian_smooth(img_t[i:i + 1])

        return img_t

# class RandGaussianSmoothD(mt.RandGaussianSmoothD):
#     def __init__(
#         self,
#         keys: KeysCollection,
#         sigma_x: tuple[float, float] = (0.25, 1.5),
#         sigma_y: tuple[float, float] = (0.25, 1.5),
#         sigma_z: tuple[float, float] = (0.25, 1.5),
#         approx: str = 'erf',
#         prob: float = 0.1,
#         *args,
#         prob_per_channel: float = 0.5,
#         **kwargs,
#     ) -> None:
#         super().__init__(keys, sigma_x, sigma_y, sigma_z, approx, prob, *args, **kwargs)
#         self.rand_smooth = RandGaussianSmooth(sigma_x, sigma_y, sigma_z, 1, approx, prob_per_channel=prob_per_channel)
