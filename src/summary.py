
import pickle
import csv
import os
import re
import sys

# out_dir_result = sys.argv[2]

# Load your list of dictionaries
# with open(data_path, "rb") as f:
#     data_list = pickle.load(f)  # this should be your list of 20 dictionaries


# print(data_list[3])


def extract_metadata(pickle_filename):
    basename = os.path.basename(pickle_filename)
    match = re.match(r"(\d+_\d+)-rot-(\d+)-crp-(\d+)", basename)
    if not match:
        raise ValueError(f"Filename {basename} doesn't match expected pattern.")
    return match.group(1), match.group(2), match.group(3)

def load_existing_rows(csv_file):
    existing = set()
    if os.path.exists(csv_file):
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Create a tuple of identifying information
                key = (row['imagename'], row['rot'], row['crop'], row['dict_number'], row['metric'])
                existing.add(key)
    return existing

def append_new_rows(pickle_file, csv_file):
    # Extract metadata
    imagename, rot, crop = extract_metadata(pickle_file)

    # Load new data
    with open(pickle_file, "rb") as f:
        data_list = pickle.load(f)

    # Load existing keys
    existing_keys = load_existing_rows(csv_file)

    # Prepare new rows
    new_rows = []
    for idx, entry in enumerate(data_list):
        for metric in ['iou', 'precision', 'recall', 'f1']:
            key = (imagename, rot, crop, str(idx), metric)
            if key not in existing_keys:
                new_rows.append({
                    'imagename': imagename,
                    'rot': rot,
                    'crop': crop,
                    'dict_number': idx,
                    'metric': metric,
                    'value': entry[metric]
                })
                

    # Append to CSV
    write_header = not os.path.exists(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['imagename', 'rot', 'crop', 'dict_number', 'metric', 'value'])
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)

    print(f"Appended {len(new_rows)} new rows to {csv_file}.")

# Example usage:
# pickle_file = "00000032_037-rot-0-crp-0-result.pkl"
csv_file = "aggregated_results.csv"
# append_new_rows(data_path, csv_file)


with open("/home/sogandzk/projects/def-dolatab6/sogandzk/CLIP-medical-domain/src/file_list.txt", "r") as f:
    paths = [line.strip() for line in f.readlines()]


for path in paths:
    try:
        append_new_rows(path, csv_file)
    except Exception as e:
        print(f"Error processing {path}: {e}")




