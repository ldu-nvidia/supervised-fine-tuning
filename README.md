Supervised-Fine-Tuning: Fine Tune foundation Model for Domain and Task Specific Applications (GTC2025)

**DLI Course ID:** x-fx-81-v1

* Author: Leo Du ldu@nvidia.com
* Maintainer: Josh Wyatt jwyatt@nvidia.com

---

**Hardware Requirements**

---

Since supervised fine tuning adjust all parameters of the model, it is a compute intensive job. This playbook can run on a compute node with at least 2 GPUs for parallel computing purposes, in the case you run into out of memory error, try to reduce global and micro batch size

---

**Software Version**

---

This tutorial uses container `nemo:24.12` please make sure you are able to pull this container from `nvcr.io/nvidia/nemo24:12`

---

**Walk through**

---

**Data Curation Step**

Before doing SFT, high quality domain specific data is needed. In this section, we will walk you through steps from downloading domain specific open source data, preprocessing, filter and deduplicate the data to form training dataset for SFT, the related codes are `curate_data.sh` which install dependency to run `curate_data.py` and the data curation pipeline is composed of the following steps. **Note the data curation use DASK data science framework which curate data in parallel on GPU, greatly increase the throughput.

* Step 1: download dataset with link specified in `sources/huggingface_urls.jsonl`
* Step 2: run curation pipeline which is composed of

  * cleans and unifies given dataset using a set of predefined cleaners
  * filter out low quality entries by a series of predefined rules, the rules are slightly differnt between text and code which include additional step of removing personaly identifiable information, filters common to text and code include:
    * word count
    * percentage of top n-gram
    * percentage of data composed of URLs
  * deduplicate data which are exactly the same or almost the same
* Step 3: randomly shuffle the verilog code and text description pairs with the same random seed. Then separate the entire dataset into train, validation and test set of predefined ratio. In this tutorial, the ratio is 80%, 15% and 5% respectively
* Step 4: merge the verilog codes and their corresponding description into a single merged dataset ready for SFT

**NOTE**: data curation could take a long time depending on amount of data and complexity of curation pipeline. Curated data is provided inside `data/merged/` folder.

---

**Supervised Fine Tuning Step**

after curating the data used for sft, we are ready to conduct the actual sft

* Step 0: you need an API to download pretrained LLM model from huggingface hub [how to create HF hub access token](https://huggingface.co/docs/hub/en/security-tokenshttps:/) place and copy paste your token to the `$HF_TOKEN`variable in `token.env` file you created
* Step 1: convert model format into `.nemo` format
* Step 2: conduct supervised fine tuning and generated fine tuned model checkpoint, this will take a while
* Step 3 evaluate the fine tuned model using the supervised fine tuned model checkpoint

**NOTE**: a sample training log from weight and bias is provided as reference `wandb-training.png`

---

**Usage**

---

1. setup correct nvcr.io access
2. setup huggingface-hub API access key
3. run `bash curate_data.sh` to curate training data (optional, the curated data is already provided in `/data/merged`, data used in this demo is in `/data/demo` consists of train, validation and test data in nemo format)
4. run `bash run_sft.sh` it will firstly pull and run `nemo:12.24` container, after the container is running, run `run_sft.py` script to conduct actual sft training and inference. After predictions are made, call build in function to evaluate the performance metrics.
5. model checkpoint after sft will be saved in `/results` folder
