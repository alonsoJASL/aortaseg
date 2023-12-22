# src/dataset.py
import torchio
from torch.utils.data import DataLoader

def create_torchio_dataset(subjects):
    return torchio.SubjectsDataset(subjects)

def create_subjects(image_paths, mask_paths):
    subjects = []
    for image_path, mask_path in zip(image_paths, mask_paths):
        subject = torchio.Subject(
            image=torchio.Image(image_path, torchio.INTENSITY),
            mask=torchio.Image(mask_path, torchio.LABEL),
        )
        subjects.append(subject)
    return subjects

def get_dataloader(image_paths, mask_paths, batch_size):
    subjects = create_subjects(image_paths, mask_paths)
    dataset = create_torchio_dataset(subjects)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
