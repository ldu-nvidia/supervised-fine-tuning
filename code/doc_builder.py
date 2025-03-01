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

import importlib
import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import dask.dataframe as dd
import pandas as pd
from dask import compute, delayed

from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import (
    read_single_partition,
    single_partition_write_with_filename,
)


class DocumentDownloader(ABC):
    """Abstract class for downloading remote data to disk"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def download(self, url):
        pass


class DocumentIterator(ABC):
    """
    Abstract iterator class for reading in raw records that have been
    downloaded to disk
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def iterate(self, file_path):
        pass


class DocumentExtractor(ABC):
    """Abstract class for extracting text from records read from disk"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def extract(self, content):
        pass


def import_downloader(downloader_path):
    module_path, downloader_name = downloader_path.rsplit(".", 1)
    downloader_module = importlib.import_module(module_path)
    downloader_class = getattr(downloader_module, downloader_name)
    if not issubclass(downloader_class, DocumentDownloader):
        raise ValueError(
            f"Input downloader {downloader_class.__name__} "
            "must be derived from DocumentDownloader defined in "
            "nemo_curator.download.docbuilder"
        )
    return downloader_class


def import_iterator(iterator_path):
    module_path, iterator_name = iterator_path.rsplit(".", 1)
    iterator_module = importlib.import_module(module_path)
    iterator_class = getattr(iterator_module, iterator_name)
    if not issubclass(iterator_class, DocumentIterator):
        raise ValueError(
            f"Input iterator {iterator_class.__name__} "
            "must be derived from DocumentIterator "
            "defined in nemo_curator.download.docbuilder"
        )
    return iterator_class


def import_extractor(extractor_path):
    module_path, extractor_name = extractor_path.rsplit(".", 1)
    extractor_module = importlib.import_module(module_path)
    extractor_class = getattr(extractor_module, extractor_name)
    if not issubclass(extractor_class, DocumentExtractor):
        raise ValueError(
            f"Input extractor {extractor_class.__name__} "
            "must be derived from DocumentExtractor defined "
            "in nemo_curator.download.docbuilder"
        )
    return extractor_class

def _iterate_and_extract_all_contents(iterator: DocumentIterator,
                                      downloaded_file: str,
                                      extractor: DocumentExtractor,
                                      output_path: str, 
                                      output_type: str):
    
    # iterate through all keys 
    for i, key in enumerate(iterator.keys):
        records = []
        # Iterate over all records in file, save four files including code and 3x descriptions
        for item in iterator.iterate(downloaded_file):
            assert len(item) == 2 and len(item[0]) == 4 and len(item[1]) == 4, "dimensionality of iterated item is wrong!"
            ## the key and index of respective contents are reversed
            record_meta, content = item[0][-(i+1)], item[1][-(i+1)]
            # Extract the text from the record
            extracted = extractor.extract(content)
            if extracted is not None:
                text_meta, text = extracted
                if text is not None:
                    line = {
                        "text": text,
                        **text_meta,
                        **record_meta,
                    }
                    records.append(line)
        partition = pd.DataFrame(records)
        filename = os.path.basename(output_path) + "_" + key + "_out"
        output_dir = os.path.dirname(output_path) + "/GaTech-EIC"
        partition["filename"] = filename
        single_partition_write_with_filename(partition, output_dir, output_type=output_type)
    # return this partition after contents of all keys are extracted
    return partition

def _download_and_extract_single_partition(
    paths: List[Tuple[str, str]],
    downloader: DocumentDownloader,
    iterator: DocumentIterator,
    extractor: DocumentExtractor,
    output_type: str,
    keep_raw_download: bool,
    force_download: bool,
    input_meta: Union[str, dict] = None,
) -> pd.DataFrame:
    url, output_path = paths

    if os.path.exists(output_path) and not force_download:
        partition = read_single_partition(
            [output_path], backend="pandas", filetype=output_type, add_filename=True
        )
        return partition

    downloaded_file = downloader.download(url)
    print("finished downloading files, downloaded file name is: ", downloaded_file)
    return _iterate_and_extract_all_contents(iterator, downloaded_file, extractor, output_path, output_type)
    
def download_and_extract(
    dataset_name: List[str],
    output_paths: List[str],
    downloader: DocumentDownloader,
    iterator: DocumentIterator,
    extractor: DocumentExtractor,
    output_format: dict,
    output_type: str = "jsonl",
    keep_raw_download=False,
    force_download=False,
    input_meta: Union[str, dict] = None,
) -> DocumentDataset:
    """
    Downloads and extracts a dataset into a format accepted by the NeMo Curator

    Args:
      dataset_name: A list of dataset_name to download the dataset from
      output_paths: A list of paths to save the final extracted output to.
        The raw output of the downloader will be saved using the path given by downloader.download(url).
      downloader: A DocumentDownloader that handles retrieving each file from its url and saving it to storage
      iterator: A DocumentIterator that handles iterating through the downloaded file's format
      extractor: A DocumentExtractor that handles extracting the data from its raw format into text
      output_format: A dictionary mappings columns to datatypes for the fields of each datapoint after extraction.
      output_type: The file type to save the dataset as.
      keep_raw_download: Whether to keep the pre-extracted download file.
      force_download: If False, will skip processing all files in output_paths that already exist and
        directly read from them instead.
      input_meta: A dictionary or a string formatted as a dictionary, which outlines
        the field names and their respective data types within the JSONL input file.

    Returns:
      A DocumentDataset of the downloaded data
    """
    if len(dataset_name) == 0:
        raise ValueError("No dataset was provided to download")
    
    output_format = dict(sorted(output_format.items()))
    df = dd.from_map(
        _download_and_extract_single_partition,
        zip(dataset_name, output_paths),
        downloader=downloader,
        iterator=iterator,
        extractor=extractor,
        output_type=output_type,
        keep_raw_download=keep_raw_download,
        force_download=force_download,
        enforce_metadata=False,
        input_meta=input_meta,
        meta=output_format,
    )
    return DocumentDataset(df)


def batch_download(dataset_name: List[str], downloader: DocumentDownloader) -> List[str]:
    """
    Downloads all the dataset_name using the downloader in parallel
    """
    delayed_downloads = [delayed(downloader.download)(url) for url in dataset_name]

    return compute(*delayed_downloads)
