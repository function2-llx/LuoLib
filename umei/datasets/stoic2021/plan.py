from pathlib import Path

import monai

img_dir = Path('cropped')

loader = monai.transforms.Compose([
    monai.transforms.LoadImageD(['img', 'mask']),
    monai.transforms.AddChannelD(['img', 'mask']),
    monai.transforms.OrientationD(['img', 'mask'], axcodes='RAS'),
])

def process_subject(patient_id):
    data = loader({
        'img': img_dir / f'{patient_id}.nii.gz',
    })

    ret = {
        'w-o': data['img'].shape[1],
        'h-o': data['img'].shape[2],
        'd-o': data['img'].shape[3],
    }
    ret.update({
        'w-c': data['img'].shape[1],
        'h-c': data['img'].shape[2],
        'd-c': data['img'].shape[3],
        'sw': data['img_meta_dict']['spacing'][0],
        'sh': data['img_meta_dict']['spacing'][1],
        'sd': data['img_meta_dict']['spacing'][2],
    })
    _data = saver(data)
    return ret

def main():
    pass

if __name__ == '__main__':
    main()
