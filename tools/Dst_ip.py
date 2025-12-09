import pandas as pd


def pross_dst_src_ip(df, str_name):
    news_cols = df[str_name].str.split('.', expand=True)
    news_cols.columns = [str_name + "1", str_name + "2", str_name + "3", str_name + "4"]
    news_cols = news_cols.astype(int)
    filter_df_new = df.drop(columns=[str_name])
    result_new_df = pd.concat([filter_df_new, news_cols], axis=1)
    return result_new_df
