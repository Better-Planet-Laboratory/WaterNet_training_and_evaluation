from water.paths import ppaths, Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import rasterio as rio
import rioxarray as rxr
from tqdm.auto import tqdm

def get_model_filenames(data_dir: Path):

    filenames_dict = {}
    first_bboxes = list(data_dir.iterdir())
    for bbox1 in first_bboxes:
        if bbox1.is_dir():
            second_bboxes = bbox1.iterdir()
            list_of_subdirs = [bbox2.name for bbox2 in second_bboxes]
            filenames_dict[bbox1.name] = list_of_subdirs[0]  # only one nested bbox
        elif bbox1.suffix == '.tif':
            filenames_dict[bbox1.name] = ''
        else:
            pass

    all_filenames = [f'{k}/{v}' if len(v) > 0 else k for k, v in filenames_dict.items() ]

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
    binned = (arr >= threshold)
    if isinstance(binned, torch.Tensor):
        binned = binned.to(int)
    if isinstance(binned, np.ndarray):
        binned = binned.astype(int)
    return binned

def save_data(arr, save_filename):
    profile = dict(
        driver='GTiff', crs=arr.rio.crs, transform=arr.rio.transform(), dtype=arr.dtype,
        width=arr.rio.width, height=arr.rio.height, count=arr.rio.count
    )
    with rio.open(save_filename, 'w', **profile) as rio_f:
        rio_f.write(arr.to_numpy())

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
        t: {'tp': tp[i], 'fp': fp[i], 'fn': fn[i], 'tn': tn[i],
            'recall': recall[i], 'precision': precision[i], 'f1': f1[i], 'specificity': specificity[i]}
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
        mask = get_pixel_mask(obs, mask_values)
        obs_binned = obs_binned[mask]
        wn = wn[mask]

    # evaluate metrics at each threshold, essentially an ROC
    roc_df = compute_metrics_at_thresholds(obs_binned, wn)

    return roc_df

def summarize_roc(roc_df, alpha=0.10):

    out_vars = ['probability',  'f1', 'recall', 'precision', 'specificity']

    # best of each:
    f1 = roc_df.iloc[[roc_df['f1'].argmax()]][out_vars]

    # best where each is above requested ci
    ci = 1 - alpha
    recall_ci = roc_df[roc_df.recall >= ci].copy()
    max_p_at_recall_ci = recall_ci.iloc[[recall_ci['precision'].argmax()]][out_vars]
    precision_ci = roc_df[roc_df.precision >= ci].copy()
    max_r_at_precision_ci = precision_ci.iloc[[precision_ci['recall'].argmax()]][out_vars]

    # put together
    smry_df = pd.concat([f1, max_p_at_recall_ci, max_r_at_precision_ci], axis=0,
                        keys=['f1', f'best_p_at_recall{ci}', f'best_r_at_precision{ci}'], names=['best_set']).droplevel(1)

    return smry_df

def get_probability_at_settings(roc: pd.DataFrame, recall_min=None, precision_min=None):
    """
    Returns the probability at which the minimum given recall and/or precision is achieved.
    At least one, or both, of recall_min and precision_min must be given.
    If only one minimum given, returns the probability that meets this minimum and results in the highest value of the other parameter.
    If both minimums are given, returns the probability that meets both settings and results in the highest F1 score -- a balance between precision and recall.

    Parameters
    ----------
    roc: Pandas DataFrame that includes, minimally, recall and precision at probability thresholds of interest.
    recall_min: Minimum recall desired.
    precision_min: Minimum precision desired.

    Returns
    -------
    Scalar probability value that meets the requirements of a minimum given recall and/or precision.
    """

    assert(recall_min is not None or precision_min is not None)

    roc_sub = roc.copy()
    if recall_min is not None:
        roc_sub = roc_sub[roc_sub['recall'] >= recall_min]
        roc_sub.sort_values('precision', ascending=False, inplace=True)
    if precision_min is not None:
        roc_sub = roc_sub[roc_sub['precision'] >= precision_min]
        roc_sub.sort_values('recall', ascending=False, inplace=True)
    if recall_min is not None and precision_min is not None:
        if 'f1' not in roc_sub.columns:
            roc_sub['f1'] = (2 * roc_sub['recall'] * roc_sub['precision']) / (roc_sub['recall'] + roc_sub['precision'])
        roc_sub.sort_values('f1', ascending=False, inplace=True)

    if len(roc_sub) == 0:
        print('Data does not meet minimum recall and/or precision requested.')
        return None

    return roc_sub.iloc[0]['probability'].item()

def bin_rasters_at_probability(probability: float, base_dir_path: Path, save_dir_path: Path):

    # load rasters
    raster_file_names = get_model_filenames(data_dir=base_dir_path)

    for f in tqdm(raster_file_names, desc='Deploying probability threshold to rasters...'):
        raster_file = base_dir_path / f
        raster = rxr.open_rasterio(raster_file)
        profile = dict(
            driver='GTiff', crs=raster.rio.crs, transform=raster.rio.transform(), dtype=raster.dtype,
            width=raster.rio.width, height=raster.rio.height, count=raster.rio.count
        )

        # bin raster
        raster_binned = classify_data(raster, threshold=probability)

        # export binned raster
        save_file = save_dir_path / f
        save_file.parent.mkdir(parents=True, exist_ok=True)
        with rio.open(save_file, 'w', **profile) as rio_f:
            rio_f.write(raster_binned.to_numpy())


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

    # get confusion matrix + metrics on masked data for each probability threshold using validation dataset
    roc_water = return_roc(mask_values=[0] + water_classes)

    # pull out desired probability thresholds
    p_recall90 = get_probability_at_settings(roc_water, recall_min=0.9)
    p_precision90 = get_probability_at_settings(roc_water, precision_min=0.9)

    # deploy probability thresholds to test dataset, for example
    bin_rasters_at_probability(probability=p_recall90,
                               base_dir_path=ppaths.model_inputs_832 / 'output_test_data_841',
                               save_dir_path=ppaths.model_inputs_832 / 'recall90_test_data_841')
    bin_rasters_at_probability(probability=p_precision90,
                               base_dir_path=ppaths.model_inputs_832 / 'output_test_data_841',
                               save_dir_path=ppaths.model_inputs_832 / 'precision90_test_data_841')

