"""
I/O functions
"""

from pathlib import Path
from tqdm import tqdm
import pandas as pd
from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose


def parse_demo_data(data_root: Path | str) -> pd.DataFrame:
    """Parse demographic data as a dataframe"""

    if not isinstance(data_root, Path):
        data_root = Path(data_root)

    demo_list = []
    phenotype_dir = data_root / "train/phenotype"
    for case_dir in tqdm(sorted(phenotype_dir.glob("sub-stroke*")), "Reading data"):
        demo_path = (
            case_dir / f"ses-01/{case_dir.name}_ses-01_demographic_baseline.xlsx"
        )
        case_data = pd.read_excel(demo_path)
        case_data["Case"] = case_dir.name
        demo_list.append(case_data)

    demo_data = pd.concat(demo_list).drop_duplicates().reset_index(drop=True)

    # Fix values
    demo_data.loc[demo_data["Hyperlipidemia"] > 1, "Hyperlipidemia"] = 1

    # Correct column data type
    cat_cols = ["Center", "Sex"]
    bool_cols = [
        "Atrial fibrillation",
        "Hypertension",
        "Diabetes",
        "Hyperlipidemia",
        "Anticoagulation",
        "Lipid lowering drugs",
        "PAIs",
        "Wake-up",
        "In-House",
        "Referral",
    ]
    time_cols = [
        "Onset to door",
        "Alert to door",
        "Door to imaging",
        "Door to groin",
        "Door to first series",
        "Time of intervention",
        "Door to recanalization",
    ]

    for col in cat_cols:
        demo_data[col] = demo_data[col].astype("category")

    for col in bool_cols:
        demo_data[col] = demo_data[col].map({0: False, 1: True}).astype("boolean")

    for col in time_cols:
        demo_data[col] = pd.to_timedelta(demo_data[col]).dt.total_seconds() / 3600

    return demo_data


def get_dataloader(
    datalist: dict,
    key: str,
    transforms: Compose,
    batch_size: int,
    shuffle: bool = True,
    cache_rate: float = 1.0,
    num_workers: int = 1,
) -> DataLoader:
    """Get dataloader using CacheDataset"""

    dataset = CacheDataset(
        data=datalist.get(key),
        transform=transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return dataloader
