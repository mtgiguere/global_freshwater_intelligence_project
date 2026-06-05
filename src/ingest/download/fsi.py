"""Process FSI annual Excel files into a single combined CSV.

Fund for Peace publishes one Excel file per year at https://fragilestatesindex.org/excel/
Place all downloaded .xlsx files in data/raw/fragilestatesindex/ then run this script.
"""

from pathlib import Path

import pandas as pd


def process_fsi(source_dir: Path, dest_dir: Path) -> Path:
    """Combine all FSI annual Excel files into one CSV for load_fsi()."""
    xl_files = sorted(source_dir.glob("[Ff][Ss][Ii]-*.xlsx"))
    if not xl_files:
        raise FileNotFoundError(f"No FSI Excel files found in {source_dir}")

    dfs = []
    for xl_file in xl_files:
        df = pd.read_excel(xl_file)
        dfs.append(df)
        print(f"  {xl_file.name}: {len(df)} rows")

    combined = pd.concat(dfs, ignore_index=True)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "fsi_combined.csv"
    combined.to_csv(dest, index=False)
    print(f"  Combined: {len(combined)} rows -> {dest}")
    return dest


if __name__ == "__main__":
    raw_dir = Path(__file__).parents[3] / "data" / "raw"
    process_fsi(
        source_dir=raw_dir / "fragilestatesindex",
        dest_dir=raw_dir / "fragilestatesindex",
    )
