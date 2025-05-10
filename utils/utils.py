import json
from collections import Counter


def count_labels(input_path):
    with open(input_path, "r") as f:
        labels = json.load(f)

    label_counts = Counter(labels.values())

    for label, count in label_counts.items():
        print(f"{label}: {count}")
    return label_counts


def index_labels(input_path):
    with open(input_path, "r") as f:
        labels = json.load(f)

    labels = {int(k): v for k, v in labels.items()}
    unique_labels = sorted(set(labels.values()))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    encoded_labels = {k: label_to_index[v] for k, v in labels.items()}

    return encoded_labels, label_to_index
