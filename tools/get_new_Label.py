import pandas as pd
import os


def process_csv(input_file: str, target_column: str):
    df = pd.read_csv(input_file)
    df['Label_num'] = df[target_column].apply(lambda x: 0 if x == 'Benign' else 1)
    base_name, ext = os.path.splitext(input_file)
    output_file = base_name + "_processed" + ext
    df.to_csv(output_file, index=False)
    print(f"saved: {output_file}")


input_file = r'Traffic Anomaly Detection\Data\cic2018\03-01-2018.csv'
target_column = 'Label'
process_csv(input_file, target_column)
