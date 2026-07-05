
import pandas as pd
import pytest
from dataset import VehicleDataset, build_class_mapping, load_classes, save_classes


def test_build_class_mapping_is_sorted_and_1_indexed():
    mapping = build_class_mapping(["car", "bicycle", "car", "bus"])
    assert mapping == {"bicycle": 1, "bus": 2, "car": 3}


def test_save_and_load_classes_round_trip(tmp_path):
    mapping = {"car": 1, "bicycle": 2}
    path = save_classes(mapping, str(tmp_path))
    loaded = load_classes(path)
    assert loaded == mapping


def test_load_classes_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_classes(str(tmp_path / "does_not_exist.json"))


def test_vehicle_dataset_uses_shared_class_mapping_not_split_local(synthetic_dataset):
    """Regression test for the original bug: class_to_idx used to be derived from
    whatever CSV a given split loaded, so a class missing from that split's rows
    (e.g. 'bicycle' appears only in one image) would shift every other class's index.
    Passing a pre-built mapping in must keep indices stable regardless of split content.
    """
    full_df = pd.read_csv(synthetic_dataset["csv_path"], header=None)
    full_df.columns = ["image_id", "label", "xmin", "ymin", "xmax", "ymax"]
    class_to_idx = build_class_mapping(full_df["label"].tolist())

    # Split that only contains the "car" rows (drops the one bicycle image).
    car_only_csv = synthetic_dataset["tmp_path"] / "car_only.csv"
    full_df[full_df["label"] == "car"].to_csv(car_only_csv, index=False)

    dataset = VehicleDataset(str(car_only_csv), synthetic_dataset["image_dir"], class_to_idx)
    # "car" must still map to the same index it would get from the full label set.
    assert dataset.class_to_idx["car"] == class_to_idx["car"]
    assert dataset.class_to_idx == class_to_idx


def test_vehicle_dataset_item_shapes(synthetic_dataset):
    full_df = pd.read_csv(synthetic_dataset["csv_path"], header=None)
    full_df.columns = ["image_id", "label", "xmin", "ymin", "xmax", "ymax"]
    class_to_idx = build_class_mapping(full_df["label"].tolist())

    full_df.to_csv(synthetic_dataset["tmp_path"] / "all.csv", index=False)
    dataset = VehicleDataset(
        str(synthetic_dataset["tmp_path"] / "all.csv"), synthetic_dataset["image_dir"], class_to_idx
    )

    assert len(dataset) == 3
    image, target = dataset[0]
    assert image.size == (100, 100)
    assert target["boxes"].shape[1] == 4
    assert target["labels"].shape[0] == target["boxes"].shape[0]


def test_vehicle_dataset_raises_when_no_images_found(tmp_path):
    empty_dir = tmp_path / "empty_images"
    empty_dir.mkdir()
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(
        [{"image_id": "00000099", "label": "car", "xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}]
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        VehicleDataset(str(csv_path), str(empty_dir), {"car": 1})
