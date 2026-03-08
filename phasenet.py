"""
PhaseNet-based phase picking for DAS data.

This module provides functions for running PhaseNet and PhaseNet-DAS
models to detect P and S wave arrivals.
"""

import os, random
SEED = 42
os.environ['PYTHONHASHSEED']          = str(SEED)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

import sys
import numpy as np
import pandas as pd
import h5py
import torch

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic        = True
    torch.backends.cudnn.benchmark            = False
    torch.backends.cudnn.enabled              = False   # kills all cuDNN kernels
torch.use_deterministic_algorithms(True)

import seisbench.models as sbm
from obspy import Trace

from . import phasenet_utils as utils

# Initialize PhaseNet models
device_torch = "cuda" if torch.cuda.is_available() else "cpu"
ml_detector = sbm.PhaseNet(in_channels=1).from_pretrained('original', version_str='latest')
ml_detector.eval()
ml_detector.to(device_torch)
ml_detector_3d = sbm.PhaseNet(in_channels=3).from_pretrained('original', version_str='latest')
ml_detector_3d.eval()
ml_detector_3d.to(device_torch)


def runPN(arr, n_component = 1): # 1c arr = (sensor, trace); 3c arr = (sensor, component (N E Z), trace)

    waveform_features = np.zeros([arr.shape[0], 2, arr.shape[-1]])
    for i in range(len(arr)):
        with torch.no_grad():
            if n_component ==1:
                PN_model = ml_detector
                input_arr = np.vstack([arr[i:i+1]] * 3).reshape(1,3,-1)
            elif n_component ==3:
                PN_model = ml_detector_3d
                input_arr = arr[i:i+1]
            
            _x = torch.from_numpy(utils.normalize_batch(input_arr)).float().to(device_torch)
            PN_probas = PN_model(_x)
            waveform_features[i] = PN_probas[:, 1:, :].cpu().detach().numpy().astype(np.float32)
    return waveform_features


    


def runPNDAS(arr,begin_time, sample_rate = 1000, gauge_length = 10):
    def _remove_duplicate(pick, chn_length):
        filtered = pick.sort_values('phase_score', ascending=False).drop_duplicates('channel_index')
        full_index = pd.Series(range(chn_length), name='channel_index')
        merged = pd.merge(full_index.to_frame(), filtered[['channel_index', 'phase_index']], 
                          on='channel_index', how='left')
        result_array = merged.to_numpy()
        return result_array
        
    tmp_file_name = './tmp_waveform.h5'
    with h5py.File(tmp_file_name, "w") as file:
        ds = file.create_dataset("data", data = arr)
        ds.attrs["dt_s"] = 1/sample_rate
        ds.attrs["dx_m"] = gauge_length
        if hasattr(begin_time, "isoformat"):
            ds.attrs["begin_time"] = begin_time.isoformat()
        else:
            ds.attrs["begin_time"] = begin_time
    with open("tmp_files.txt", "w") as f:
        f.write(tmp_file_name)
    EQNetpath = '/home/jk103/02.Reagle_microseismic/Reagle_microseismic/EQNet'
    base_cmd = f"{EQNetpath}/predict.py --model phasenet_das --data_list=tmp_files.txt  --data_path {tmp_file_name} --result_path ./tmp_results --format=h5  --batch_size 1 --workers 0"
    cmd = f"{sys.executable} {base_cmd}"
    os.system(cmd)
    picks = pd.read_csv("./tmp_results/picks_phasenet_das/tmp_waveform.csv")
    p_pick = _remove_duplicate(picks[picks.phase_type =='P'], arr.shape[0])
    s_pick = _remove_duplicate(picks[picks.phase_type =='S'], arr.shape[0])
    return p_pick, s_pick
        
    
    


def get_picks(das_arr, threshold_P=0.2, threshold_S=0.2):
	waveform_features = runPN(das_arr)
	waveform_features.shape

	sum_prob = waveform_features.sum(axis=1)
	picks = pd.DataFrame(
	    index=np.arange(waveform_features.shape[0]),
	    columns=["P_probas", "P_picks", "P_unc", "S_probas", "S_picks", "S_unc"],
	)
	for sta in range(waveform_features.shape[0]):
	    picks.loc[sta, ["P_probas", "P_picks", "P_unc"]] = utils.find_picks(
	        waveform_features[sta, 0, :], threshold_P
	    )
	    picks.loc[sta, ["S_probas", "S_picks", "S_unc"]] = utils.find_picks(
	        waveform_features[sta, 1, :], threshold_S
	    )
	P_result = np.array([(i, p[0] if p.size > 0 else np.nan) for i, p in enumerate(picks.P_picks)])
	S_result = np.array([(i, p[0] if p.size > 0 else np.nan) for i, p in enumerate(picks.S_picks)])
	return P_result, S_result, sum_prob



def get_stalta_triggers(das_arr, sampling_rate, sta, lta, thresh_on, thresh_off):
    """
    Detects all event onsets in a DAS array using the STA/LTA algorithm.

    Parameters
    ----------
    das_arr : np.ndarray
        2D array of DAS data (channels, samples).
    sampling_rate : float
        Sampling rate of the data in Hz.
    sta : float
        Short-term average window length in seconds.
    lta : float
        Long-term average window length in seconds.
    thresh_on : float
        STA/LTA ratio threshold for trigger activation.
    thresh_off : float
        STA/LTA ratio threshold for trigger deactivation.

    Returns
    -------
    np.ndarray
        A 2D array of shape (n_picks, 2) where each row is [channel_index, pick_time].
        Picks are sorted by channel index, then by time.
    """
    all_picks = []
    for i, trace_data in enumerate(das_arr):
        trace = Trace(data=trace_data)
        trace.stats.sampling_rate = sampling_rate
        cft = classic_sta_lta(trace.data, int(sta * sampling_rate), int(lta * sampling_rate))
        on_off = trigger_onset(cft, thresh_on, thresh_off)
        
        if len(on_off) > 0:
            for onset in on_off[:, 0]:
                all_picks.append([i, onset])
            
    return np.array(all_picks)
    

