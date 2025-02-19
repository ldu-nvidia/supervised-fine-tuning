# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import os
import json
import shutil
import numpy as np
from typing import Any, Optional, Tuple
from functools import reduce
import random

import matplotlib.pyplot as plt
from downloaders import (
    download_github_sources,
    download_pdf_sources,
    download_wikipedia_sources,
    download_huggingface_sources,
)
from utils import (
    CodeLineCountFilter,
    TextLineCountFilter,
    clean_and_unify,
    dedupe,
    filter_code,
    filter_text,
    redact_code,
)

import nemo_curator as nc
from nemo_curator import ExactDuplicates, Modify, ScoreFilter, Sequential
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import RepeatingTopNGramsFilter, WordCountFilter
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.modifiers.unicode_reformatter import UnicodeReformatter
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.file_utils import (
    get_all_files_paths_under,
    separate_by_metadata,
)
from nemo_curator.utils.script_utils import ArgumentHelper

SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR_PATH, "data")

def download_sources(hf_limit: Optional[int] = None,
) -> str:
    """
    Downloads all the dataset sources and converts them to the JSONL format.
    Args:
        wikipedia_limit (int): Maximum number of wiki urls to be downloaded
        github_limit (int): Maximum number of github repos to be downloaded
        pdf_limit (int): Maximum number of pdf to be downloaded
        hf_limit (int): Maximum number of huggingface datasets to be downloaded

    Returns:
        tuple: the list of text files and the list of code files.
    """

    hf_dir = download_huggingface_sources("sources/huggingface_urls.jsonl", limit = hf_limit)
    hf_files = get_all_files_paths_under(hf_dir)

    # delete original file once extracted into four files
    code_file = ""
    text_file = []
    for file in hf_files:
        if "code_out.jsonl" in file:
            code_file += file
        elif "MG-Verilog.jsonl" in file:
            os.remove(file)
            print("removed original download")
        else:
            text_file.append(file)
    return text_file, code_file

def run_curation_pipeline(args: Any, text_files: list, code_files: str) -> list:
    """
    Run the curation pipeline on the verilog dataset.

    Args:
        args (Any): Command-line arguments.
        jsonl_dir (str): Directory path where the JSONL files are stored.
    """
    print("Running the curation pipeline...")
    # Initialize the Dask cluster.
    client = get_client(**ArgumentHelper.parse_client_args(args))
    
    # Overwrite existing files in the curated directory.
    out_path = os.path.join(DATA_DIR, "curated")
    if os.path.isdir(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    # Define data curation steps for text and pdf files
    curation_steps_text = Sequential(
        [
            clean_and_unify,
            ScoreFilter(
                TextLineCountFilter(), text_field="file_type_count", score_type=bool
            ),
            filter_text,
            dedupe,
        ]
    )

    # Define data curation steps for code files
    curation_steps_code = Sequential(
        [
            clean_and_unify,
            ScoreFilter(
                CodeLineCountFilter(), text_field="file_type_count", score_type=bool
            ),
            filter_code,
            dedupe,
            redact_code,
        ]
    )
    # keep record of all the entries that pass curation, only keep the entries of the unions of them
    all_passed_ids = []
    for text_file in text_files:
        orig_dataset_text = DocumentDataset.read_json(text_file, add_filename=True)
        # create a field combining fields file type and line count
        orig_dataset_text.df["file_type_count"] = (
            orig_dataset_text.df["file_type"]
            + " : "
            + orig_dataset_text.df["line_count"].astype(str)
        )
        dataset_text = curation_steps_text(orig_dataset_text)
        # execute the data curation
        dataset_text = dataset_text.persist()
        dataset_text.to_json(out_path, write_to_filename=True)
        print(f"Original dataset length for text files: {len(orig_dataset_text.df)}")
        print(f"After data curation: {len(dataset_text.df)}")
        all_passed_ids.append(dataset_text.df["id"].values.compute())
    print("finish curating text files!")
    
    # curate code
    print("start curating code")
    orig_dataset_code = DocumentDataset.read_json(code_files, add_filename=True)
    orig_dataset_code.df["file_type_count"] = (
        orig_dataset_code.df["file_type"]
        + " : "
        + orig_dataset_code.df["line_count"].astype(str)
    )
    dataset_code = curation_steps_code(orig_dataset_code)
    dataset_code = dataset_code.persist()
    dataset_code.to_json(out_path, write_to_filename=True)
    print(f"Original dataset length for code files: {len(orig_dataset_code.df)}")
    print(f"Length of code dataset after curation: {len(dataset_code.df)}")

    all_passed_ids.append(dataset_code.df["id"].values.compute())
    selected_ids = np.array(reduce(np.intersect1d, tuple(all_passed_ids)))
    print("length of selected entries: ", len(selected_ids))
    client.close()
    return selected_ids

def write_file(data, filename):
    assert data and len(data) != 0, "data provided is empty, did not save!"
    with open(filename, "w") as f:
        for line in data:
            f.write(line.strip() + "\n")
    print("finished writing file: ", filename)


def merge_in_output(text_files: list, code_files: list):
    # create a new folder to save merged results
    original_text_files = copy.deepcopy(text_files)
    out_path = os.path.join(DATA_DIR, "merged/")
    if os.path.isdir(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)
    for i in range(len(text_files)):
        for j in range(len(text_files[i])):
            assert "split" in text_files[i][j], "file name is wrong"
            text_files[i][j] = text_files[i][j].replace("split", "merged").replace("out", "in_out")

    for desc_idx in range(len(original_text_files)):
        for tvt_idx in range(len(original_text_files[desc_idx])):
            with open(original_text_files[desc_idx][tvt_idx],"r") as txt, open(code_files[tvt_idx], "r") as code:
                with open(text_files[desc_idx][tvt_idx], "w") as result:
                    for one_text, one_code in zip(txt, code):
                        input = one_text.split(',"id":')[0]
                        output = one_code.split(',"id":')[0]
                        id_input = int(one_text.split(',"id":')[1].split(',"file_extension"')[0])
                        id_output = int(one_text.split(',"id":')[1].split(',"file_extension"')[0])
                        assert id_input == id_output, "id of text is different from id of code!"
                        new_entry = {"input": input, "output": output, "id": id_input}
                        result.write(json.dumps(new_entry) + "\n")
                    print("finished merging: ", text_files[desc_idx][tvt_idx], '\n')
                    result.close()
                txt.close()
                code.close()
    print("finished entire data preparation process!!")

# list of text files to be splitted, code file string to be splitted
# then filterout the discarded entries, seed make sure random shuffle is the same for multiple shuffles
def split_datasets(text_files: list, code_files: str, selected_ids: list, seed: int):
    out_path = os.path.join(DATA_DIR, "split/")
    if os.path.isdir(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    split_text_dirs, split_code_dirs = [], None
    # split all text files
    for text_file in text_files:
        training_output_file = out_path + text_file.rsplit("/")[-1][:-6] + "_train.jsonl"
        validation_output_file = out_path + text_file.rsplit("/")[-1][:-6] +  "_validation.jsonl"
        test_output_file = out_path + text_file.rsplit("/")[-1][:-6] + "_test.jsonl"
        split_text_dirs.append([training_output_file, validation_output_file, test_output_file])

        # specify proportion of data for training and validation
        train_proportion = 0.80
        validation_proportion = 0.15
        assert train_proportion > 0.0 and validation_proportion > 0.0 and train_proportion + validation_proportion < 1.0, "either train or validation proportion is not right!"

        # read and shuffle JSON file objects
        with open(text_file, "r") as f:
            training_output_file = out_path + text_file.rsplit("/")[-1][:-6] + "_train.jsonl"
            validation_output_file = out_path + text_file.rsplit("/")[-1][:-6] +  "_validation.jsonl"
            test_output_file = out_path + text_file.rsplit("/")[-1][:-6] + "_test.jsonl"
            lines = f.readlines()
            # strip out the id of line and only keep the selected ids
            lines = [line for line in lines if int(line.split('"id":')[1].split(',"file_extension"')[0]) in selected_ids]
            assert len(lines) == len(selected_ids), "number of curated text data is not right!"
            random.seed(seed)
            random.shuffle(lines)
            # calculate split indices
            total_lines = len(lines)
            train_index = int(total_lines * train_proportion)
            val_index = int(total_lines * (train_proportion + validation_proportion))

            # distribute JSON objects into train, validation, tests sets
            train_data = lines[:train_index]
            validation_data = lines[train_index:val_index]
            test_data = lines[val_index:]

            # write JSON objects to files
            write_file(train_data, training_output_file)
            write_file(validation_data, validation_output_file)
            write_file(test_data, test_output_file)
            print("finish splitting text train, validation and test data")

    # next split code files
    with open(code_files, "r") as f:
        lines = f.readlines()
        training_output_file = out_path + code_files.rsplit("/")[-1][:-6] + "_train.jsonl"
        validation_output_file = out_path + code_files.rsplit("/")[-1][:-6] +  "_validation.jsonl"
        test_output_file = out_path + code_files.rsplit("/")[-1][:-6] + "_test.jsonl"
        split_code_dirs = [training_output_file, validation_output_file, test_output_file]
        # strip out the id of line and only keep the selected ids
        lines = [line for line in lines if int(line.split('"id":')[1].split(',"file_extension"')[0]) in selected_ids]
        assert len(lines) == len(selected_ids), "number of curated code data is not right!"
        random.seed(seed)
        random.shuffle(lines)
        train_data = lines[:train_index]
        validation_data = lines[train_index:val_index]
        test_data = lines[val_index:]
        # write JSON objects to files
        write_file(train_data, training_output_file)
        write_file(validation_data, validation_output_file)
        write_file(test_data, test_output_file)
        print("finish splitting code train, validation and test data")
    return (split_text_dirs, split_code_dirs)

def delete_intermediate_data():
    ## to save space
    shutil.rmtree(os.path.join(DATA_DIR, 'raw/'))
    shutil.rmtree(os.path.join(DATA_DIR, 'curated/'))
    shutil.rmtree(os.path.join(DATA_DIR, 'split/'))
    print("deleted all intermediate files to save space")

def main():
    parser = argparse.ArgumentParser()
    args = ArgumentHelper(parser).add_distributed_args().parse_args()
    # Limit the total number of workers to ensure we don't run out of memory.
    args.n_workers = min(args.n_workers, 8)
    print("Args: ", args)
    # Download all the sources and get the list of text and code files.
    text_files, code_file = download_sources(1)
    # curate the text and code data given quality filters
    selected_ids = run_curation_pipeline(args, text_files, code_file)
    # shuffle the data and create train, val, test sets
    split_text_dirs, split_code_dirs = split_datasets(text_files, code_file, selected_ids, 0)
    # merge description as input and code as output for SDG applications
    merge_in_output(split_text_dirs, split_code_dirs)
    # delete unnecessary files to save space
    delete_intermediate_data()


if __name__ == "__main__":
    main()