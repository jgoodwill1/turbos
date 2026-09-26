import os
import numpy as np
import pandas as pd
from cdasws import CdasWs

# ----------------------------
# constants
# ----------------------------
MU0 = 4e-7 * np.pi
M_P = 1.6726219e-27  # kg

cdas = CdasWs()

OUTDIR = "psp_E1_E7_10s"
os.makedirs(OUTDIR, exist_ok=True)

# ----------------------------
# datasets / main variables
# ----------------------------
MAG_DATASET = "PSP_FLD_L2_MAG_RTN_4_SA_PER_CYC"
MAG_VAR = "psp_fld_l2_mag_RTN_4_Sa_per_Cyc"
MAG_TIME = "epoch_mag_RTN_4_Sa_per_Cyc"

SPC_DATASET = "PSP_SWP_SPC_L3I"
QTN_DATASET = "PSP_FLD_L3_RFS_LFR_QTN"
SPAN_DATASET = "PSP_SWP_SPI_SF00_L3_MOM"

# Added for PSP radial distance
POS_DATASET = "PSP_SWP_SPC_L3I"
POS_VAR = "sc_pos_HCI"

# ----------------------------
# YOU fill in your encounter windows here
# ----------------------------
encounters = [
    ("E1",  "2018-10-31", "2018-11-11"),
    ("E2",  "2019-03-30", "2019-04-10"),
    ("E3",  "2019-08-16", "2019-09-20"),
    ("E4",  "2020-01-23", "2020-02-29"),
    ("E5",  "2020-05-09", "2020-06-28"),
    ("E6",  "2020-09-21", "2020-10-03"),
    ("E7",  "2021-01-12", "2021-01-24"),
    # ("E8",  "2021-04-24", "2021-05-06"),
    # ("E9",  "2021-08-01", "2021-08-16"),
    # ("E10", "2021-11-14", "2021-11-28"),
    # ("E11", "2022-02-18", "2022-03-03"),
    # ("E12", "2022-05-24", "2022-06-07"),
    # ("E13", "2022-08-30", "2022-09-12"),
    # ("E14", "2022-12-05", "2022-12-18"),
    # ("E15", "2023-03-10", "2023-03-24"),
    # ("E16", "2023-06-15", "2023-06-29"),
    # ("E17", "2023-09-20", "2023-10-04"),
    # ("E18", "2023-12-22", "2024-01-05"),
    # ("E19", "2024-03-23", "2024-04-06"),
    # ("E20", "2024-06-23", "2024-07-07"),
    # ("E21", "2024-09-23", "2024-10-07"),
    # ("E22", "2024-12-17", "2024-12-31"),
    # ("E23", "2025-03-15", "2025-03-29"),
    # ("E24", "2025-06-12", "2025-06-26"),
    # ("E25", "2025-09-08", "2025-09-22"),
]

# ----------------------------
# helpers
# ----------------------------
def find_time_coord(ds):
    for c in ds.coords:
        if "epoch" in c.lower():
            return c
    raise KeyError(f"No epoch coordinate found. coords={list(ds.coords)}")

def first_existing(options, names):
    for name in options:
        if name in names:
            return name
    return None

def print_vars(dataset):
    print(f"\n=== {dataset} ===")
    for v in cdas.get_variables(dataset):
        print(v["Name"])

def autodetect_names():
    spc_vars = [v["Name"] for v in cdas.get_variables(SPC_DATASET)]
    qtn_vars = [v["Name"] for v in cdas.get_variables(QTN_DATASET)]
    span_vars = [v["Name"] for v in cdas.get_variables(SPAN_DATASET)]

    spc_density = first_existing(
        ["np1_fit_gd", "np1_fit", "np_fit", "np", "Np"],
        spc_vars
    )
    spc_flag = first_existing(
        ["general_flag", "GENERAL_FLAG", "dqf", "DQF"],
        spc_vars
    )

    qtn_density = first_existing(
        ["N_elec"],
        qtn_vars
    )
    qtn_low = first_existing(
        ["N_elec_deltalow"],
        qtn_vars
    )
    qtn_high = first_existing(
        ["N_elec_deltahigh"],
        qtn_vars
    )

    span_vel = first_existing(
        ["VEL_RTN_SUN"],
        span_vars
    )
    span_flag = first_existing(
        ["QUALITY_FLAG", "quality_flag", "general_flag", "DQF"],
        span_vars
    )

    return {
        "spc_density": spc_density,
        "spc_flag": spc_flag,
        "qtn_density": qtn_density,
        "qtn_low": qtn_low,
        "qtn_high": qtn_high,
        "span_vel": span_vel,
        "span_flag": span_flag,
    }

names = autodetect_names()
print(names)

if names["spc_density"] is None:
    raise RuntimeError("Could not auto-detect SPC density variable. Run print_vars(SPC_DATASET).")
if names["qtn_density"] is None:
    raise RuntimeError("Could not auto-detect QTN density variable. Run print_vars(QTN_DATASET).")
if names["span_vel"] is None:
    raise RuntimeError("Could not auto-detect SPAN-I RTN Sun-frame velocity variable. Run print_vars(SPAN_DATASET).")

# ----------------------------
# source downloaders
# each returns a 10 s dataframe
# ----------------------------
def download_mag_day(t0, t1):
    status, ds = cdas.get_data(MAG_DATASET, [MAG_VAR], t0, t1)
    if ds is None:
        return None

    t = pd.to_datetime(ds[MAG_TIME].values)
    B = ds[MAG_VAR].values

    df = pd.DataFrame(B, index=t, columns=["Br", "Bt", "Bn"])
    df["Bmag_nT"] = np.sqrt(df["Br"]**2 + df["Bt"]**2 + df["Bn"]**2)
    df["Bmag_T"] = df["Bmag_nT"] * 1e-9

    return df.resample("10s").mean()

def download_spc_day(t0, t1):
    req = [names["spc_density"]]
    if names["spc_flag"] is not None:
        req.append(names["spc_flag"])

    status, ds = cdas.get_data(SPC_DATASET, req, t0, t1)
    if ds is None:
        return None

    tname = find_time_coord(ds)
    t = pd.to_datetime(ds[tname].values)

    df = pd.DataFrame(index=t)
    df["n_spc_cm3"] = ds[names["spc_density"]].values

    if names["spc_flag"] is not None:
        df["spc_flag"] = ds[names["spc_flag"]].values
        df = df[df["spc_flag"] == 0]

    # basic physical mask
    df.loc[df["n_spc_cm3"] <= 0, "n_spc_cm3"] = np.nan

    return df.resample("10s").mean()

def download_qtn_day(t0, t1):
    req = [names["qtn_density"]]
    if names["qtn_low"] is not None:
        req.append(names["qtn_low"])
    if names["qtn_high"] is not None:
        req.append(names["qtn_high"])

    status, ds = cdas.get_data(QTN_DATASET, req, t0, t1)
    if ds is None:
        return None

    tname = find_time_coord(ds)
    t = pd.to_datetime(ds[tname].values)

    df = pd.DataFrame(index=t)
    df["n_qtn_cm3"] = ds[names["qtn_density"]].values

    if names["qtn_low"] is not None:
        df["n_qtn_low_cm3"] = ds[names["qtn_low"]].values
    if names["qtn_high"] is not None:
        df["n_qtn_high_cm3"] = ds[names["qtn_high"]].values

    # basic masks for fill/unphysical values
    df.loc[df["n_qtn_cm3"] <= 0, "n_qtn_cm3"] = np.nan
    df.loc[df["n_qtn_cm3"] < -1e20, "n_qtn_cm3"] = np.nan

    return df.resample("10s").mean()

def download_span_day(t0, t1):
    req = [names["span_vel"]]
    if names["span_flag"] is not None:
        req.append(names["span_flag"])

    status, ds = cdas.get_data(SPAN_DATASET, req, t0, t1)
    if ds is None:
        return None

    tname = find_time_coord(ds)
    t = pd.to_datetime(ds[tname].values)

    vel = ds[names["span_vel"]].values
    # assume RTN component order [R, T, N]
    # verify once using ds and metadata in your environment
    df = pd.DataFrame(index=t)
    df["Vr_kms"] = vel[:, 0]
    df["Vt_kms"] = vel[:, 1]
    df["Vn_kms"] = vel[:, 2]
    # if names["span_flag"] is not None:
    #     flag = ds[names["span_flag"]].values
    #     # handle either scalar or array-like flag
    #     if np.ndim(flag) == 1 and len(flag) == len(df):
    #         df["span_flag"] = flag
    #         # keep nominal if 0 is available
    #         df = df[(df["span_flag"] == 0) | (df["span_flag"].isna())]

    # # basic physical mask
    df.loc[~np.isfinite(df["Vr_kms"]), "Vr_kms"] = np.nan
    df = df.resample("10s").mean()
    print(df["Vr_kms"])
    return df

def download_pos_day(t0, t1):
    status, ds = cdas.get_data(POS_DATASET, [POS_VAR], t0, t1)
    if ds is None:
        return None

    tname = find_time_coord(ds)
    t = pd.to_datetime(ds[tname].values)

    pos = ds[POS_VAR].values  # HCI position in km, shape (N, 3)

    df = pd.DataFrame(index=t)
    df["r_km"] = np.sqrt((pos ** 2).sum(axis=1))

    R_SUN_KM = 695700.0
    AU_KM = 1.495978707e8

    df["r_Rsun"] = df["r_km"] / R_SUN_KM
    df["r_AU"] = df["r_km"] / AU_KM

    return df.resample("10s").mean()

# ----------------------------
# per-encounter pipeline
# ----------------------------
def process_encounter(enc_name, start_date, end_date):
    print(f"\n===== {enc_name}: {start_date} to {end_date} =====")

    days = pd.date_range(start_date, end_date, freq="1D")

    mag_parts = []
    spc_parts = []
    qtn_parts = []
    span_parts = []
    pos_parts = []

    for i in range(len(days) - 1):
        t0 = days[i].strftime("%Y-%m-%dT00:00:00Z")
        t1 = days[i+1].strftime("%Y-%m-%dT00:00:00Z")
        print(" ", t0, "->", t1)

        try:
            x = download_mag_day(t0, t1)
            if x is not None:
                mag_parts.append(x)
        except Exception as e:
            print("   MAG failed:", e)

        try:
            x = download_spc_day(t0, t1)
            if x is not None:
                spc_parts.append(x)
        except Exception as e:
            print("   SPC failed:", e)

        try:
            x = download_qtn_day(t0, t1)
            if x is not None:
                qtn_parts.append(x)
        except Exception as e:
            print("   QTN failed:", e)

        try:
            x = download_span_day(t0, t1)
            if x is not None:
                span_parts.append(x)
        except Exception as e:
            print("   SPAN-I failed:", e)

        try:
            x = download_pos_day(t0, t1)
            if x is not None:
                pos_parts.append(x)
        except Exception as e:
            print("   POS failed:", e)

    if not mag_parts:
        print("No MAG data found for", enc_name)
        return None

    df_mag = pd.concat(mag_parts).sort_index()
    df_spc = pd.concat(spc_parts).sort_index() if spc_parts else pd.DataFrame(index=df_mag.index)
    df_qtn = pd.concat(qtn_parts).sort_index() if qtn_parts else pd.DataFrame(index=df_mag.index)
    df_span = pd.concat(span_parts).sort_index() if span_parts else pd.DataFrame(index=df_mag.index)
    df_pos = pd.concat(pos_parts).sort_index() if pos_parts else pd.DataFrame(index=df_mag.index)

    # merge on the 10 s grid
    df = df_mag.join(df_spc, how="left").join(df_qtn, how="left").join(df_span, how="left").join(df_pos, how="left")
    df = df[~df.index.duplicated(keep="first")].sort_index()

    # densities: cm^-3 -> m^-3
    df["n_spc_m3"] = df["n_spc_cm3"] * 1e6
    df["n_qtn_m3"] = df["n_qtn_cm3"] * 1e6

    # mass densities (quasi-neutral approximation for QTN)
    df["rho_spc"] = df["n_spc_m3"] * M_P
    df["rho_qtn"] = df["n_qtn_m3"] * M_P

    # Alfven speed from |B|
    df["Va_spc_mps"] = df["Bmag_T"] / np.sqrt(MU0 * df["rho_spc"])
    df["Va_qtn_mps"] = df["Bmag_T"] / np.sqrt(MU0 * df["rho_qtn"])

    # radial speed from SPAN-I
    df["Vr_mps"] = np.abs(df["Vr_kms"]) * 1e3

    # radial Alfven Mach numbers
    df["Ma_r_spc"] = df["Vr_mps"] / df["Va_spc_mps"]
    df["Ma_r_qtn"] = df["Vr_mps"] / df["Va_qtn_mps"]

    # optional masks
    bad = (~np.isfinite(df["Bmag_T"])) | (df["Bmag_T"] <= 0)
    for col in ["Ma_r_spc", "Ma_r_qtn", "Va_spc_mps", "Va_qtn_mps"]:
        df.loc[bad, col] = np.nan

    # save outputs
    base = os.path.join(OUTDIR, enc_name)
    df.to_parquet(base + "_merged_10s.parquet")
    df_mag.to_parquet(base + "_mag_10s.parquet")
    if len(df_spc) > 0:
        df_spc.to_parquet(base + "_spc_10s.parquet")
    if len(df_qtn) > 0:
        df_qtn.to_parquet(base + "_qtn_10s.parquet")
    if len(df_span) > 0:
        df_span.to_parquet(base + "_spani_10s.parquet")

    print(f"Saved {enc_name}: {len(df):,} rows")
    return df

# ----------------------------
# run all encounters
# ----------------------------
all_dfs = {}
for enc_name, start_date, end_date in encounters:
    out = process_encounter(enc_name, start_date, end_date)
    if out is not None:
        all_dfs[enc_name] = out