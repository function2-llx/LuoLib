from pathlib import Path

import SimpleITK as sitk
import pandas as pd
from tqdm.contrib.concurrent import process_map

def process(patient_id: str):
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(data_dir / f'{patient_id}.mha'))   # Give it the mha file as a string
    reader.LoadPrivateTagsOn()     # Make sure it can get all the info
    reader.ReadImageInformation()  # Get just the information from the file
    age = reader.GetMetaData('PatientAge')
    sex = reader.GetMetaData('PatientSex')
    return int(age[:-1]), int(sex == 'M')


origin_dir = Path('origin')
data_dir = origin_dir / 'data' / 'mha'

def main():
    ref = pd.read_csv(origin_dir / 'metadata' / 'reference.csv')
    ref['age'], ref['sex'] = zip(*process_map(process, ref['PatientID'], chunksize=1))
    ref.to_csv('reference.csv', index=False)

if __name__ == '__main__':
    main()
