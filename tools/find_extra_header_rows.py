import csv


def find_extra_header_rows(file_path, encoding='utf-8'):
    extra_rows = []
    with open(file_path, newline='', encoding=encoding) as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return extra_rows
        for lineno, row in enumerate(reader, start=2):
            if row == header:
                extra_rows.append(lineno)

    return extra_rows


if __name__ == "__main__":
    path = r"Traffic Anomaly Detection\Data\cic2019\Syn.csv"
    repeats = find_extra_header_rows(path)
    if repeats:
        print("Duplicate header rows detected at the following line numbers:")
        for ln in repeats:
            print(f"  Line {ln}")
    else:
        print("No duplicate header rows detected.")

