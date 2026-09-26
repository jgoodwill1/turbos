import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import importlib
from scipy.stats import gaussian_kde, multivariate_normal
import scipy.constants as c
import encs as E
importlib.reload(E)

encs = np.arange(1,7)

for enc in encs:
    print(enc)
    fname = f"psp_E1_E7_10s/E{enc}_merged_10s.parquet"
    df = pd.read_parquet(fname)
    print(df.keys())

    cols = [
        'Br', 'Bt', 'Bn', 'Bmag_nT', 'Bmag_T', 'n_spc_cm3', 'spc_flag',
       'n_qtn_cm3', 'n_qtn_low_cm3', 'n_qtn_high_cm3', 'Vr_kms', 'Vt_kms',
       'Vn_kms', 'r_km', 'r_Rsun', 'r_AU', 'n_spc_m3', 'n_qtn_m3', 'rho_spc',
       'rho_qtn', 'Va_spc_mps', 'Va_qtn_mps', 'Vr_mps', 'Ma_r_spc',
       'Ma_r_qtn'
    ]
    df = df[cols]
    # df['VR'] = df_read['VR'].reindex(df.index, method="nearest")
    # optional: drop rows where everything is missing
    df = df.dropna(how="all")
    # print(df)

    df["n_merged_cm3"] = df["n_spc_cm3"].combine_first(df["n_qtn_cm3"])

    both = df["n_spc_cm3"].notna() & df["n_qtn_cm3"].notna()

    ratio = df["n_spc_cm3"] / df["n_qtn_cm3"]
    bad_agreement = both

    df.loc[bad_agreement, "n_merged_cm3"] = np.nan
    df.loc[bad_agreement, "density_source"] = "disagree"

    df["n_comb_cm3"] = df["n_spc_cm3"].combine_first(df["n_qtn_cm3"])
    M_P = 1.6726219e-27  # kg
    MU0 = 4e-7 * np.pi
    df["n_spc_m3"] = df["n_spc_cm3"] * 1e6
    df["n_qtn_m3"] = df["n_qtn_cm3"] * 1e6
    df["n_comb_m3"] = df["n_comb_cm3"] * 1e6
    df["rho_spc"] = df["n_spc_m3"] * M_P
    df["rho_qtn"] = df["n_qtn_m3"] * M_P
    df["rho_comb"] = df["n_comb_m3"] * M_P
    df['VA_qtn'] = df['Bmag_nT']/np.sqrt(MU0 * df['rho_qtn']) * 1e-9 * 1e-3
    df['VA_spc'] = df['Bmag_nT']/np.sqrt(MU0 * df['rho_spc']) * 1e-9 * 1e-3
    df['VA_comb'] = df['Bmag_nT']/np.sqrt(MU0 * df['rho_comb']) * 1e-9 * 1e-3

    df['MA_spc'] = df['Vr_kms']/df['VA_spc']
    df['MA_qtn'] = df['Vr_kms']/df['VA_qtn']
    df['MA_comb'] = df['Vr_kms']/df['VA_comb']
    
    df.to_parquet(f"psp_E1_E7_10s_comb/E{enc}_comb_10s.parquet")