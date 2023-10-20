from monai.data import set_track_meta, get_track_meta

__all__ = [
    'track_meta',
]

class track_meta:
    def __init__(self, value: bool):
        self.origin = get_track_meta()
        set_track_meta(value)

    def __exit__(self, *args, **kwargs):
        set_track_meta(self.origin)
