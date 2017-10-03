from utils import dataset_helper

all_datasets = dataset_helper.get_all_available_dataset_names()

def get_dataset_from_filename(filename: str) -> str:
    filename = filename.split('/')[-1]
    candidates = sorted([dataset for dataset in all_datasets if dataset in filename])
    return candidates[-1] if len(candidates) else None
