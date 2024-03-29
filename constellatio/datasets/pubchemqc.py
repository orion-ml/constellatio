import os
import csv

import requests
from tempfile import gettempdir

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer
import selfies as sf
import random

from tqdm import tqdm

TOTAL_AMOUNT_CIDS = 17309040


def get_cids(limit, offset=None):
    """
    Generator function to retrieve cids from the API.
    """
    try:
        if offset is None:
            offset = random.randint(0, TOTAL_AMOUNT_CIDS - limit)
        base_url = "https://pcqc.matter.toronto.edu/pm6opt_chon300nosalt"
        payload = {
            "select": "cid",
            "order": "cid.asc",
            "limit": limit,
            "offset": offset,
        }

        response = requests.get(base_url, params=payload)
        response.raise_for_status()
        cids = response.json()

        for entry in cids:
            yield entry["cid"]

    except requests.RequestException as e:
        print(f"An network error occurred: {e}")
    except ValueError:
        print("Response is not a valid JSON")
    except Exception as e:
        print(f"An error occurred: {e}")


def fetch_molecule_data(cids, chunk_size=200):
    base_url = "https://pcqc.matter.toronto.edu/pm6opt_chon300nosalt"
    molecule_data_list = []

    for i in range(0, len(cids), chunk_size):
        chunk_cids = cids[i:i+chunk_size]
        cid_list = f"({','.join(map(str, chunk_cids))})"
        params = {"select": "*", "cid": f"in.{cid_list}"}

        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            molecule_data_list.extend(response.json())
        else:
            print(f"Failed to fetch data for cids {chunk_cids}: {response.status_code}")

    return molecule_data_list


def generate_dataset_from_ids(ids, data_path=None, output_file=None, selfies=True, test_size=0.2, batch_size=500):
    """
    Generates a dataset from a list of cids.

    :param ids: List of cids to fetch data for.
    :param data_path: The base directory to use for writing data files.
    :param output_file: Optional path to the output CSV file.
    :param selfies: Boolean indicating whether to use SELFIES representation.
    :param test_size: Fraction of the dataset to be used as the test set.
    :param batch_size: Number of cids to process in each batch.
    """

    if data_path is None:
        data_path = gettempdir()

    temp_dir = os.path.join(data_path, "orion_temp")
    os.makedirs(temp_dir, exist_ok=True)

    output_file = output_file or os.path.join(temp_dir, "pubchemqc_dataset.csv")

    ids_list = list(ids)
    total_ids = len(ids_list)

    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([f'{"selfies" if selfies else "smiles"}', "energy"])

        for i in tqdm(range(0, total_ids, batch_size), desc="Processing Batches"):
            batch_ids = ids_list[i:i + batch_size]
            
            molecule_data = fetch_molecule_data(batch_ids)
            for data in molecule_data:
                if data:
                    try:
                        molecule_rep = data["data"]["pubchem"]["Isomeric SMILES"]
                        energy = data["data"]["pubchem"]["PM6"]["properties"]["energy"]["total"]
                        if selfies:
                            molecule_rep = sf.encoder(molecule_rep)
                        writer.writerow([molecule_rep, energy])

                    except KeyError:
                        continue

    # Split the dataset and save to disk
    dataset_split_path = os.path.join(temp_dir, "pubchemqc_dataset_split")
    data = Dataset.from_csv(output_file)
    data.train_test_split(test_size=test_size).save_to_disk(dataset_split_path)

    return dataset_split_path

def get_partition_bounds(client_id, n_clients):
    """
    Calculate the starting and ending indices of a partition.
    """
    partition_size, remainder = divmod(TOTAL_AMOUNT_CIDS, n_clients)
    start_index = partition_size * client_id
    end_index = (
        start_index + partition_size + (remainder if client_id == n_clients - 1 else 0)
    )

    return start_index, end_index


def partition_cids(client_id, n_clients, max_data=None, output_file=None):
    """
    Partitions the cids into n_clients partitions and returns the partition for client_id.
    Optionally limits the number of cids fetched by the max_data parameter.
    """
    start_index, end_index = get_partition_bounds(client_id, n_clients)

    required_cids = end_index - start_index
    if max_data is not None:
        required_cids = min(required_cids, max_data)

    current_cid_count = 0

    def cid_generator():
        nonlocal current_cid_count, required_cids, start_index
        while current_cid_count < required_cids:
            for cid in get_cids(
                limit=min(200, required_cids - current_cid_count),
                offset=start_index + current_cid_count,
            ):
                yield cid
                current_cid_count += 1
                if max_data is not None and current_cid_count >= max_data:
                    return

    return cid_generator()


# def random_cids(n_cids):
#     """
#     Returns n_cids random CIDs.
#     """
#     if n_cids > TOTAL_AMOUNT_CIDS:
#         raise ValueError("n_cids cannot be greater than TOTAL_AMOUNT_CIDS")

#     random_indices = np.random.choice(TOTAL_AMOUNT_CIDS, n_cids, replace=False)
#     random_indices.sort()

#     random_cids_result = []
#     for index in random_indices:
#         cid_data = list(get_cids(limit=1, offset=index))
#         if cid_data:
#             random_cids_result.append(cid_data[0])

#     return random_cids_result

# def random_cids(n_cids):
#     randoms_cids_result = []
#     while len(randoms_cids_result) < n_cids:
#         cid_data = list(get_cids(limit=200))
#         if cid_data:
#             randoms_cids_result.append(cid_data)
#     return randoms_cids_result


def generate_dataset_partition(
    client_id, n_clients, max_data=None, output_path=None, selfies=True, data_path=None
):
    """
    Generates a dataset partition for a given client_id and n_clients and returns the path to the dataset.
    """
    ids = partition_cids(client_id, n_clients, max_data)
    return generate_dataset_from_ids(ids, output_file=output_path, selfies=selfies, data_path=data_path)


def generate_dataset(n_data, output_path=None, selfies=True, data_path=None):
    """
    Generates a dataset of size n_data and returns the path to the dataset.

    :param n_data: The number of data points to generate.
    :param output_path: Optional path to the output CSV file.
    :param selfies: Boolean indicating whether to use SELFIES representation.
    :param data_path: The base directory to use for writing data files.
    """
    ids = get_cids(n_data)
    return generate_dataset_from_ids(ids, output_file=output_path, selfies=selfies, data_path=data_path)


def load_data(
    path, tokenizer: PreTrainedTokenizer, tokenizer_kwargs=None, loader_kwargs=None
):
    """
    Loads the dataset from a given path and creates data loaders.

    :param path: Path where the dataset is stored.
    :param tokenizer: Tokenizer for processing the data.
    :param tokenizer_kwargs: Additional kwargs for tokenizer (optional).
    :param loader_kwargs: Default kwargs for all dataloaders (optional).
    :return: Tuple of DataLoaders for the train, validation, and test sets.
    """

    default_tokenizer_kwargs = {
        "truncation": True,
        "max_length": 35,
        "return_tensors": "pt",
        "padding": True,
    }

    default_loader_kwargs = {
        "batch_size": 4,
        "shuffle": True,
        # 'num_workers': 1,  # Uncomment if necessary
    }

    if tokenizer_kwargs is not None:
        default_tokenizer_kwargs.update(tokenizer_kwargs)

    if loader_kwargs is not None:
        default_loader_kwargs.update(loader_kwargs)

    tokenizer_kwargs = default_tokenizer_kwargs
    loader_kwargs = default_loader_kwargs

    dataset = DatasetDict.load_from_disk(path)
    trainset, testset = dataset["train"], dataset["test"]

    trainset_length = int(len(trainset) * 0.8)
    lengths = [trainset_length, len(trainset) - trainset_length]
    trainset, valset = random_split(trainset, lengths)

    def my_collator(examples):
        """Custom collation function for DataLoader."""
        output = tokenizer(
            [
                e["selfies"].replace("]", "] ") for e in examples
            ],  # Add space after each bracket to use Marko's tokenizer
            **tokenizer_kwargs,
        )
        output["labels"] = torch.tensor([e["energy"] for e in examples])
        return output

    def create_loader(dataset):
        """Utility function to create a DataLoader."""
        return DataLoader(dataset, collate_fn=my_collator, **loader_kwargs)

    # Create data loaders
    train_loader = create_loader(trainset)
    val_loader = create_loader(valset)
    test_loader = create_loader(testset)

    return train_loader, val_loader, test_loader


def chunked(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate dataset for a client.")

    parser.add_argument(
        "-c",
        "--client_id",
        type=int,
        help="The id of the client to generate data for.",
        default=0,
    )
    parser.add_argument(
        "-n",
        "--n_clients",
        type=int,
        help="The total number of clients to generate data for.",
        default=1,
    )

    parser.add_argument(
        "-md",
        "--max_data",
        type=int,
        help="The maximum number of data points to generate.",
    )

    parser.add_argument(
        "-nd",
        "--n_data",
        type=int,
        help="The total number of data points to generate.",
        default=0,
    )

    args = parser.parse_args()
    client_id = args.client_id
    n_clients = args.n_clients
    n_data = args.n_data
    max_data = args.max_data or None

    if n_data == 0:
        print("Generating dataset partition...")
        generate_dataset_partition(client_id, n_clients, max_data)

    else:
        print(f"Generating data with size {n_data}...")
        generate_dataset(n_data)