import pandas as pd


def split_flow_id(df: pd.DataFrame) -> pd.DataFrame:
    """
Split the 'Flow ID' column in the DataFrame by '-' into five new columns:
    'Flow ID1' (source IP), 'Flow ID2' (destination IP), 'Flow ID3' (source port),
    'Flow ID4' (destination port), 'Flow ID5' (protocol number), and return the expanded DataFrame.

Parameters
----------
df : pandas.DataFrame
    Original DataFrame containing the 'Flow ID' column.

Returns
-------
pandas.DataFrame
    DataFrame with five new split columns added to the original df.
"""

    if 'Flow ID' not in df.columns:
        raise KeyError("DataFrame can not find 'Flow ID'")
    parts = df['Flow ID'].str.split('-', expand=True)
    if parts.shape[1] != 5:
        raise ValueError(f"Split 'Flow ID' column into {parts.shape[1]} segments, expected 5 segments")
    parts.columns = ['Flow ID1', 'Flow ID2', 'Flow ID3', 'Flow ID4', 'Flow ID5']
    for col in ['Flow ID3', 'Flow ID4', 'Flow ID5']:
        parts[col] = parts[col].astype(int)
    return pd.concat([df, parts], axis=1)
