def parse_dataset_name(name: str) -> tuple[str, str]:
    parts = name.split('_', 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid format: '{name}'. Expected 'prefix_suffix'.")
    prefix, suffix = parts
    return prefix, f"{suffix}.csv"


if __name__ == "__main__":
    prefix, filename = parse_dataset_name("cic2019_LDAP")
    print(prefix)    # cic2019
    print(filename)  # LDAP.csv
