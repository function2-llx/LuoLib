from typing import Optional

from matplotlib import pyplot as plt
import numpy as np

class IndexTracker:
    def __init__(self, img: np.ndarray, seg: Optional[np.ndarray] = None):
        fig, ax = plt.subplots()
        fig: plt.Figure
        ax: plt.Axes
        ax.set_title('use scroll wheel to navigate images')
        self.ax = ax

        self.img = img
        self.seg = seg
        rows, cols, self.slices = img.shape
        self.ind = self.slices//2

        self.ax_img = ax.imshow(np.rot90(self.img[:, :, self.ind]), cmap='gray')
        if self.seg is None:
            self.ax_seg = None
        else:
            self.ax_seg = ax.imshow(np.rot90(self.seg[:, :, self.ind]), cmap='gray')

        self.update()
        fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        plt.show()

    def on_scroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = min(self.img.shape[-1] - 1, self.ind + 1)
        else:
            self.ind = max(0, self.ind - 1)
        self.update()

    def update(self):
        self.ax.set_ylabel('slice %s' % self.ind)
        self.ax_img.set_data(np.rot90(self.img[:, :, self.ind]))
        self.ax_img.axes.figure.canvas.draw()
        if self.ax_seg is not None:
            self.ax_seg.set_data(np.rot90(self.seg[:, :, self.ind]))
            self.ax_seg.axes.figure.canvas.draw()
