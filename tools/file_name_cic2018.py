import re


def parse_cic_filename(filename: str):
    match = re.match(r"(cic\d{4})_(\d{2})_(\d{2})_(\d{4})", filename)
    if match:
        prefix = match.group(1)
        date_str = f"{match.group(2)}-{match.group(3)}-{match.group(4)}.csv"
        return prefix, date_str
    else:
        raise ValueError("The input format does not meet the expected structure, for example:'cic2018_02_14_2018'")


if __name__ =="__main__":
    name, csv_name = parse_cic_filename("cic2018_02_14_2018")
    print(name)  # cic2018
    print(csv_name)  # 02-14-2018.csv
