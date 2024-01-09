import pandas as pd

def edit_pred_file(trial: int, mode: str, model: str, values, val_pred: str):
    if not mode in ["train", "test"]:
        raise ValueError("mode must be either \"train\" or \"test\"")
    elif not model in ["control", "mlr", "mlp"]:
        raise ValueError("model must be either control mlr or mlp")
    elif not val_pred in ["area", "x", "y"]:
        raise ValueError("bad val_pred")
    pred_file = pd.read_csv(f"data/raw-data/predictions/trial_{trial}_{mode}.csv", index_col=0)
    if not len(values) == len(pred_file.index):
        raise ValueError(f"values needs to be of lengths {len(pred_file.index)} not {len(values)}")
    pred_file[f"{model}_{val_pred}_pred"] = values

    pred_file.to_csv(f"data/raw-data/predictions/trial_{trial}_{mode}.csv")