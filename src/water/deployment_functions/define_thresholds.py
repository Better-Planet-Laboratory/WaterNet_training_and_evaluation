from water.paths import ppaths, Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import rasterio as rio
from tqdm.auto import tqdm

def get_model_filenames(data_dir: Path):

    filenames_dict = {}
    first_bboxes = data_dir.iterdir()
    for bbox1 in first_bboxes:
        second_bboxes = bbox1.iterdir()
        list_of_subdirs = [bbox2.name for bbox2 in second_bboxes]
        filenames_dict[bbox1.name] = list_of_subdirs[0]  # only one nested bbox

    all_filenames = [f'{k}/{v}' for k, v in filenames_dict.items()]

    return all_filenames

def load_model_dataset(list_of_files):

    ds = []
    for f in list_of_files:
        with rio.open(f) as rio_f:
            data = rio_f.read()
        data = torch.from_numpy(data)
        if data.shape[0] != 1:
            data = data.unsqueeze(0)
        ds.append(data)
    return torch.cat(ds)

def harmonize_data(arr):
    return F.max_pool2d(arr.unsqueeze(0), kernel_size=2).squeeze(0)

def classify_data(arr, threshold=0.5):
    return (arr >= threshold).to(int)

def get_pixel_mask(data: torch.Tensor, included_values: list[int]):
    return torch.isin(data, torch.tensor(included_values))

def compute_metrics_at_thresholds(targets: torch.Tensor, probabilities: torch.Tensor):
    inputs_np = probabilities.detach().cpu().float().numpy().flatten()
    targets_np = targets.detach().cpu().float().numpy().flatten()

    # Sort by probability descending
    sort_idx = np.argsort(-inputs_np)
    inputs_sorted = inputs_np[sort_idx]
    targets_sorted = targets_np[sort_idx].astype(np.int32)

    # Append np.inf to capture boundary where classifier always predicts negative class
    inputs_sorted = np.append([np.inf], inputs_sorted)

    # Cumulative TP and FP as we lower the threshold
    tp_cumsum = np.append([0], np.cumsum(targets_sorted))
    fp_cumsum = np.append([0], np.cumsum(1 - targets_sorted))

    total_pos = targets_np.sum(dtype=np.int64)
    total_neg = len(targets_np) - total_pos

    # Unique threshold indices
    threshold_idxs = np.where(np.diff(inputs_sorted, append=np.inf))[0]

    thresholds =  inputs_sorted[threshold_idxs]
    tp = tp_cumsum[threshold_idxs]
    fp = fp_cumsum[threshold_idxs]
    fn = total_pos - tp
    tn = total_neg - fp

    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1 = (2 * tp) / ((2 * tp) + fp + fn)

    metrics_list = {
        t: {'recall': recall[i], 'precision': precision[i], 'f1': f1[i], 'specificity': specificity[i]}
        for i, t in enumerate(thresholds)
    }

    # make into dataframe
    metrics_df = pd.DataFrame.from_dict(metrics_list, orient='index').rename_axis('probability').reset_index()
    metrics_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return metrics_df

def return_roc(inputs_dir: Path =  ppaths.model_inputs_832 / 'val_data',
       outputs_dir: Path = ppaths.model_inputs_832 / 'output_val_data_841',
       mask_values: list[int] = []):

    # load data
    filenames = get_model_filenames(data_dir=inputs_dir)
    obs_files = [inputs_dir / f / 'waterways_burned.tif' for f in filenames]
    wn_files = [outputs_dir / f for f in filenames]

    obs = harmonize_data(load_model_dataset(obs_files))
    wn = load_model_dataset(wn_files)

    # categorize observed data
    obs_binned = classify_data(obs, threshold=1)

    # mask data, if requested
    if len(mask_values) > 0:
        mask = get_pixel_mask(obs_binned, mask_values)
        obs_binned = obs_binned[mask]
        wn = wn[mask]

    # evaluate metrics at each threshold, essentially an ROC
    roc_df = compute_metrics_at_thresholds(obs_binned, wn)

    return roc_df



if __name__ == '__main__':
    mult = 2.0
    ww_value_dict = {
        1: 0.0,  # playa
        2: 0.0,  # Inundation
        3: 0.5,  # Swamp I
        4: 0.5,  # Swamp P
        5: 0.5,  # Swamp
        6: mult * 1.0,  # Reservoir
        7: 0.5,  # Lake I
        8: mult * 3.5,  # Lake P
        9: mult * 3.5,  # Lake
        10: 0.0,  # spillway
        11: 0.5,  # drainage
        12: 0.5,  # wash
        13: 0.5,  # canal storm
        14: 1.0,  # canal aqua
        15: 0.5,  # canal
        16: 1.0,  # artificial path
        17: mult * 3.75,  # Ephemeral
        18: mult * 3.75,  # Intermittent
        19: mult * 3.25,  # Perennial
        20: mult * 3.25,  # streams
        21: 1.0,  # other
    }
    water_classes = [k for k, v in ww_value_dict.items() if v >= 1]
