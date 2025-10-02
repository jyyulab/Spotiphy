import warnings
import json
import matplotlib.image as mpimg
import os
import scanpy as sc
import pandas as pd


def _load_visium_hd_metadata(outs_dir, bin_size_um, spatial_dir):
    """
    Load images and scalefactors_json.json for Visium HD into a dict structure.
    Returns a dict keyed by one "library_id" (here 'visium_hd').
    """
    library_id = "visium_hd"

    # 1) Load scalefactors JSON (if present)
    scalefactors_path = os.path.join(spatial_dir, "scalefactors_json.json")
    scalefactors = {}
    if os.path.exists(scalefactors_path):
        with open(scalefactors_path, "r") as f:
            scalefactors = json.load(f)

    # 2) Load images if available
    images = {}
    for res in ["hires", "lowres"]:
        image_name = f"tissue_{res}_image.png"
        fpath = os.path.join(spatial_dir, image_name)
        if os.path.exists(fpath):
            try:
                images[res] = mpimg.imread(fpath)
            except Exception:
                images[res] = None

    # 3) Collect into a dictionary mimicking Scanpy/Squidpy style
    meta = {
        "images": images,  # dict of numpy arrays
        "scalefactors": scalefactors,  # dict from scalefactors_json.json
        "bin_size_um": int(bin_size_um),
        "spatial_dir": spatial_dir,
    }

    return {library_id: meta}


def load_visium_hd_to_anndata(
    outs_dir,
    bin_size_um=8,
    positions_basename="tissue_positions",
    prefer_coord=("x", "y"),
    fallback_coord_order=(
        ("x_global", "y_global"),
        ("x_local", "y_local"),
        ("pxl_row_in_fullres", "pxl_col_in_fullres"),
    ),
    check_unique_barcodes=True,
):
    """Load Visium HD Spaceranger outputs into an AnnData using only scanpy + pandas + os.path.

    Args:
        outs_dir (str): Path to the Spaceranger `outs/` directory for the HD run.
        bin_size_um (int): Bin size (e.g., 8 or 16). Used to locate the 'square_XXXum' folder.
        positions_basename (str): Base filename for the binned positions table (without extension).
            This function will try '<basename>.parquet' first, then '<basename>.csv'.
        prefer_coord (tuple[str, str]): Column names to use first for coordinates (default: ("x", "y")).
        fallback_coord_order (tuple[tuple[str, str], ...]): Ordered fallbacks for coordinate columns if `prefer_coord` not found.
        check_unique_barcodes (bool): If True, emit a warning when duplicate barcodes appear in the positions table.

    Returns:
        anndata.AnnData: Counts + coordinates in `adata.obsm["spatial"]`. The positions table
            is merged into `adata.obs`. Minimal metadata in `adata.uns["spatial"]`.
    """
    # 1) Resolve bin folder like 'binned_outputs/square_008um/spatial'
    #    zero-pad bin size to 3 digits to match 10x convention
    square_dir = os.path.join(
        outs_dir, "binned_outputs", f"square_{int(bin_size_um):03d}um"
    )
    spatial_dir = os.path.join(square_dir, "spatial")
    if not os.path.isdir(spatial_dir):
        # try alternate common location (some runs differ), else error
        alt_dir = os.path.join(outs_dir, square_dir, "spatial")
        if os.path.isdir(alt_dir):
            spatial_dir = alt_dir
        else:
            raise FileNotFoundError(
                "Could not find spatial positions directory at:\n"
                f"  {spatial_dir}\n  or {alt_dir}\n"
                "Please check bin_size_um or your outs_dir structure."
            )

    # 2) Read counts
    counts_h5 = os.path.join(square_dir, "filtered_feature_bc_matrix.h5")
    if not os.path.exists(counts_h5):
        raise FileNotFoundError(f"Could not find counts file: {counts_h5}")
    adata = sc.read_10x_h5(counts_h5)

    # 3) Read positions: try parquet, then csv
    parquet_path = os.path.join(spatial_dir, f"{positions_basename}.parquet")
    csv_path = os.path.join(spatial_dir, f"{positions_basename}.csv")
    if os.path.exists(parquet_path):
        pos = pd.read_parquet(parquet_path)
        src_ext = ".parquet"
    elif os.path.exists(csv_path):
        pos = pd.read_csv(csv_path)
        src_ext = ".csv"
    else:
        raise FileNotFoundError(
            "Could not find positions file as parquet or csv:\n"
            f"  {parquet_path}\n  {csv_path}"
        )

    # 4) Basic hygiene: expected barcode column
    if "barcode" not in pos.columns:
        cand = [c for c in pos.columns if "barcode" in c.lower()]
        if cand:
            pos = pos.rename(columns={cand[0]: "barcode"})
        else:
            raise ValueError(
                f"No 'barcode' column found in {positions_basename}{src_ext}. "
                f"Columns were: {list(pos.columns)}"
            )

    if check_unique_barcodes and pos["barcode"].duplicated().any():
        dups = pos["barcode"][pos["barcode"].duplicated()].unique()[:5]
        warnings.warn(
            f"Duplicate barcodes found in positions table (showing first 5): {dups}. "
            "Will deduplicate by keeping the first occurrence."
        )
        pos = pos.drop_duplicates(subset="barcode", keep="first")

    pos = pos.set_index("barcode")

    # 5) Align to adata barcodes
    #    (10x barcodes usually match exactly; if you have suffixes like '-1' adjust here)
    missing = [b for b in adata.obs_names if b not in pos.index]
    if len(missing) > 0:
        print(
            f"{len(missing)} barcodes in counts were not found in positions. "
            "Trying to strip suffixes after '-' to match."
        )
        stripped = [b.split("-")[0] for b in adata.obs_names]
        if len(set(stripped)) == adata.n_obs and all(s in pos.index for s in stripped):
            adata.obs_names = stripped
            missing = []
        else:
            warnings.warn(
                f"{len(missing)} barcodes in counts were not found in positions. "
                "Those observations will have NaN coordinates."
            )

    # Merge all pos columns into obs (preserve any metadata there)
    adata.obs = adata.obs.join(pos, how="left")

    # 6) Choose coordinate columns
    def pick_coords(df, prefer_xy, fallbacks):
        if all(c in df.columns for c in prefer_xy):
            return prefer_xy
        for pair in fallbacks:
            if all(c in df.columns for c in pair):
                return pair
        # last resort: guess any two columns that look like x/y
        xcands = [c for c in df.columns if c.lower().startswith("x")]
        ycands = [c for c in df.columns if c.lower().startswith("y")]
        if xcands and ycands:
            return (xcands[0], ycands[0])
        raise ValueError(
            "Could not find coordinate columns. "
            f"Tried {prefer_xy} and fallbacks {fallbacks}."
        )

    xcol, ycol = pick_coords(adata.obs, prefer_coord, fallback_coord_order)

    # 7) Put coordinates into .obsm['spatial'] (float)
    coords = adata.obs[[ycol, xcol]].astype("float64").to_numpy()
    adata.obsm["spatial"] = coords

    # 8) Minimal metadata in .uns['spatial'] for downstream tools
    adata.uns["spatial"] = _load_visium_hd_metadata(
        outs_dir=outs_dir, bin_size_um=bin_size_um, spatial_dir=spatial_dir
    )

    # 9) Report basic alignment stats
    n_with_coords = int(pd.notnull(adata.obs[xcol]).sum())
    if n_with_coords < adata.n_obs:
        warnings.warn(
            f"Only {n_with_coords}/{adata.n_obs} observations have coordinates. "
            "Check barcode matching or positions file."
        )

    return adata
