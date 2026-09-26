import os
import sys

import numpy as np
import pandas as pd
import scipy.constants as c

sys.path.append("/bigdata/space_plasma/DATA/PSP_encounters/scripts/")
from get_datetimes_from_encs import get_fname

DATA_DIR = "/bigdata/space_plasma/DATA/PSP_encounters"

# ============================================================
# 1. LOW-LEVEL I/O
# ============================================================


def _load_encounter_csv(enc, res):
    fname = os.path.join(DATA_DIR, get_fname(enc, res=res))
    df = pd.read_csv(
        fname,
        sep="\t",
        names=["t", "BR", "BT", "BN", "VR", "VT", "VN", "nSP", "nSQ", "nQ"],
    )
    df["t"] = pd.to_datetime(df["t"], origin="unix", unit="s")
    return df.set_index("t").sort_index()


def load_density(enc, index):
    enc_date = pd.read_csv(os.path.join(DATA_DIR, "encounters.csv")).set_index(
        "encounter"
    )
    s = enc_date.loc[enc]["start_date"]
    e = enc_date.loc[enc]["end_date"]

    fname = os.path.join(
        DATA_DIR, "combined_density", f"density_E{enc:02d}_{s}_{e}_60s_unix.csv"
    )

    den = pd.read_csv(fname, names=["t", "ne", "np", "n"], skiprows=1)
    den["t"] = pd.to_datetime(den["t"], origin="unix", unit="s")
    den = den.set_index("t").sort_index()
    
    target_index = pd.DatetimeIndex(index)

    # include both original density times and desired output times
    full_index = den.index.union(target_index).sort_values()

    # interpolate in time, then return only the requested times
    n_interp = (
        den["n"]
        .reindex(full_index)
        .interpolate(method="time", limit_area="inside")
        .reindex(target_index)
    )
    
    return n_interp * 1e6


def load_R(index, res):
    R = pd.read_csv(os.path.join(DATA_DIR, f"R_20180813_20240413_{res}s.csv"))
    R["t(UTC)"] = pd.to_datetime(R["t(UTC)"], origin="unix", unit="s")
    R = R.set_index("t(UTC)").sort_index()
    if "R(r_sun)" in R:
        R = R.rename(columns={"R(r_sun)": "R"})
    return R["R"].reindex(index, method="nearest")


def load_all_encounters(enc_range, res=60, avg=None):
    parts = []
    for enc in enc_range:
        try:
            df = _load_encounter_csv(enc, res)
            df["n"] = load_density(enc, df.index)
            df["encounter"] = enc
            if avg:
                df = df.resample(f"{avg}").mean().dropna()
            parts.append(df)
        except Exception as e:
            print(f"Encounter {enc} failed:", e)
    return pd.concat(parts).sort_index()


# ============================================================
# 2. CORE FIELDS
# ============================================================


def Bmag(df):
    return np.sqrt(df["BR"] ** 2 + df["BT"] ** 2 + df["BN"] ** 2)


def Vmag(df):
    return np.sqrt(df["VR"] ** 2 + df["VT"] ** 2 + df["VN"] ** 2)


def VA(df, avg_win="60s"):
    mask = (df["n"] > 0) & (df["n"] < 1e15)
    B_T = Bmag(df[mask]) * 1e-9
    return (B_T / np.sqrt(c.mu_0 * c.proton_mass * df.loc[mask, "n"])).rolling(avg_win, center = True).mean() * 1e-3


def M_A(df):
    V_A = VA(df)
    VR = df["VR"]
    return VR / V_A


# ============================================================
# 3. TURBULENCE
# ============================================================


def Z(df, mean_window="6h"):
    B = df[["BR", "BT", "BN"]]
    Bm = B.rolling(mean_window, center=True).mean()
    dot = (B * Bm).sum(axis=1)
    mag = np.sqrt((B**2).sum(axis=1))
    magm = np.sqrt((Bm**2).sum(axis=1))
    return 0.5 * (1 - (dot / (mag * magm)).clip(-1, 1))


def dB_comp(df, avg="60s", norm=False):
    df[["BR", "BT", "BN"]] = df[["BR", "BT", "BN"]] * 1e-9
    # Bmean = df[['BR','BT','BN']].resample(avg).mean()
    Bmean = df[["BR", "BT", "BN"]].rolling(avg, center=True).mean()
    df2 = df[["BR", "BT", "BN"]] ** 2
    # B2mean = df2.resample(avg).mean()
    B2mean = df2.rolling(avg, center=True).mean()
    Bmean["dBR"] = B2mean["BR"] - Bmean["BR"] ** 2
    Bmean["dBT"] = B2mean["BT"] - Bmean["BT"] ** 2
    Bmean["dBN"] = B2mean["BN"] - Bmean["BN"] ** 2
    Bmean["dB"] = Bmean["dBR"] + Bmean["dBT"] + Bmean["dBN"]
    if norm:
        Bmean["dB"] = Bmean["dB"] * 1e8 / c.mu_0
    return Bmean[["dBR", "dBT", "dBN", "dB"]]


def dB(df, roll="60s", avg="60s", norm=False):
    Bmag = np.sqrt(df["BR"] ** 2 + df["BT"] ** 2 + df["BN"] ** 2)
    B0 = Bmag.rolling(roll, center=True).mean()
    b = Bmag - B0
    dB = np.sqrt((b**2).resample(avg).mean()).dropna()
    if norm:
        dB = dB / Bmag.resample(avg).mean()
    return dB


def dVR(df, roll_win="60s", avg_win="60s", norm=False):
    VR0 = df["VR"].rolling(roll_win, center=True).mean()
    v = df["VR"] - VR0
    dVR = np.sqrt((v**2).resample(avg_win).mean())
    if norm:
        dVR = dVR / VA(df).resample(avg_win).mean()
    return dVR


# ============================================================
# 4. SUB-ALFVÉNIC INTERVALS
# ============================================================


def subalf(df, min_duration="10min"):
    ma = M_A(df)
    sub = ma < 1
    group = (sub != sub.shift()).cumsum()
    dt = ma.index.to_series().diff().median()
    dur = sub.groupby(group).sum() * dt
    good = dur[dur > pd.Timedelta(min_duration)].index
    return group.isin(good)


# ============================================================
# 5. INCREMENTS
# ============================================================


def increments(df, tau_s, norm=True):
    dt = df.index.to_series().diff().median().total_seconds()
    k = int(round(tau_s / dt))
    df["VA"] = VA(df)

    Bmag = np.sqrt(df["BR"] ** 2 + df["BT"] ** 2 + df["BN"] ** 2)
    Bmag_fwd = Bmag.shift(-k)

    Bf = df[["BR", "BT", "BN", "VR", "VA"]].shift(-k)

    dVR = np.abs((Bf["VR"] - df["VR"]) / np.abs((Bf["VA"] + df["VA"]) / 2))
    dBR = np.abs(Bf["BR"] - df["BR"]) / np.abs((Bmag + Bmag_fwd) / 2)

    return pd.DataFrame({"dBR": dBR, "dVR": dVR}, index=df.index)


# ============================================================
# 6. EQUAL-N BINNING
# ============================================================


def bin_average(x, y, N=100):
    df = pd.DataFrame({"x": x, "y": y}).dropna().sort_values("x")
    df["bin"] = np.arange(len(df)) // N
    return df.groupby("bin").agg(
        x_center=("x", "mean"),
        y_median=("y", "median"),
        y_mean=("y", "mean"),
        y_std=("y", "std"),
    )


def fill_frac(x, y, N=100, z_thresh=0.5):
    mask = y > z_thresh
    x = x[mask]
    y = y[mask]

    # --- log bins
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    x_min = max(x_min, 1e-3)

    bins = np.logspace(np.log10(x_min), np.log10(x_max), N + 1)

    # --- bin
    x_bin = pd.cut(x, bins=bins)

    # --- count per bin
    out = (
        pd.Series(y)
        .groupby(x_bin, observed=True)
        .count()
        .rename("n_sb")
        .reset_index(name="n_sb")
    )

    out.columns = ["x_bin", "n_sb"]

    # --- fractions
    out["frac"] = out["n_sb"] / out["n_sb"].sum()
    out["nsum"] = out["n_sb"].cumsum()
    out["cumsum"] = out["frac"].cumsum()

    # --- bin centers
    out["x_center"] = out["x_bin"].apply(lambda iv: iv.mid)

    return out
