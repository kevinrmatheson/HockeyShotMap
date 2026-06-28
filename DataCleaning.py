import pandas as pd

DEFAULT_INPUT_CSV = "2018NHLShotInfoV2.csv"


def load_shot_data(csv_path: str = DEFAULT_INPUT_CSV) -> pd.DataFrame:
	# Minimal loading helper for analysis notebooks and quick data checks.
	"""Load shot event data from a CSV export.

	This utility is intentionally small for now and can be extended as
	cleaning rules are defined.
	"""

	return pd.read_csv(csv_path)

