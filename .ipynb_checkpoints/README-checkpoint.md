# kairos-data-curation
Quarter 2 Project for DSC180AB

The objective of this project is to develop and evaluate a KAIROS-based pipeline for data valuation and curated fine-tuning, applied to an image classification task.

* Our large “messy” dataset is the **iNaturalist dataset**, containing relevant samples of our target classes (insect species) as well as irrelevant data: https://github.com/visipedia/inat_comp/tree/master/2021
* Our clean dataset is the **Kaggle Insects Image dataset** with the desired insect images: https://www.kaggle.com/datasets/ismail703/insects/data

The kairos_inat_valuation.ipynb notebook requires a Python *3.9* environment

```bash
# Move to the Kairos directory to run the relevant files
cd utils

# Step 1: Install required packages from requirements.txt
python3 -m pip install -r kairos_requirements.txt

# Step 2: Install the benchmark tool opendataval
python3 -m pip install --no-dependencies opendataval

# Step 3: Fix the data-loading bug in the package by running the overwrite script
python3 overwrite_package.py
```

embedding_extractor.ipynb requires a newer python version 3.11.9
python3 -m pip install -r emb_ex_requirements.txt



### Project Structure
```
└── kairos-data-curation/
    ├── data/                         # where all data files and generated embeddings are stored
        ├── clean_insect_images/      # clean validation dataset from Kaggle
            ├── Ant/                  # Ant images
            ├── Bee/                  # Bee images
            └── ...                   # More images of different insect species organized into folders by class
        └── embs/                     # Contains generated embeddings, labels, filepaths, and indexes
    ├── EDA/                          # Preliminary files
        ├── eda.ipynb                 # Exploring the clean Kaggle dataset
        └── overlap_asessment.ipynb   # Assessing dataset sizes and overlap
    ├── utils/                        # Label maps and KAIROS functions
        ├── otdd/                     # Optimal transport dataset distance
            ├── pytorch/              # Distance functions
            ├── plotting.py           # Plotting funtions
            └── utils.py              # Supporting functions
        ├── custom_valuations.py      # KAIROS class
        ├── label_mapping.py          # Includes iNat species to class mapping in iNat_to_clean_map
        ├── overwrite_package.py      # Overwrites bug in opendataval
        ├── requirements.txt          # KAIROS dependencies
        └── sample_clean_data.py      # Gets stratified random sample of clean data to create the KAIROS validation set
    ├── final_resnet.ipynb                  # Experiment for fine-tuning ResNet50 model via LoRA and partial freezing methods using different fine-tuning datasets
    ├── embedding_extractor.ipynb     # Generates image embeddings for iNat and clean data: inat_embs/ and clean_embs/
    ├── kairos_inat_valuation.ipynb   # Uses KAIROS to curate iNat images for fine-tuning ResNet based on clean data. Will generate curated_embs.npy               
    └── README.md

```
