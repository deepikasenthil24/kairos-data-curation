# Data Valuation for Curated Fine-Tuning using KAIROS
Quarter 2 Project for DSC180AB

**Website:** https://deepikasenthil24.github.io/efficient-fine-tuning-data-curation-kairos/


This repository implements a data curation pipeline using KAIROS to identify high-utility samples from "noisy" datasets, allowing for more efficient model training. We demonstrate this by curating the iNaturalist dataset to improve the fine-tuning of a ResNet-50 model on a target Insects classification task.


### A. Problem Description
Modern machine learning models like **ResNet-50** rely on massive datasets for fine-tuning. However, real-world datasets like **iNaturalist** are often "messy," containing poisoned, mislabeled, or noisy samples that degrade model performance and trustworthiness.

Training on these full datasets is computationally expensive and inefficient. ML engineers currently lack scalable, automated methods to distinguish between high-value samples that drive performance and low-quality data that introduces feature interference. This project addresses the need for a **model-agnostic data valuation** approach, using **KAIROS** to curate high-utility subsets. By identifying the most impactful data points, we maintain model performance while achieving a **93.6% reduction in training time.**


### B. Dataset Context
* Large “messy” dataset: The [**iNaturalist dataset**](https://github.com/visipedia/inat_comp/tree/master/2021), containing relevant samples of our target classes (insect species) as well as significant amounts of irrelevant or "noisy" data.
* Clean validation dataset: The [**Kaggle Insects Image dataset**](https://www.kaggle.com/datasets/ismail703/insects/data), which contains the desired high-quality insect images used to define our target distribution.


### C. Repository Organization
<!-- 
```
└── kairos-data-curation/
    ├── data/                         # Where all data files and generated embeddings are stored
    |    ├── clean_insect_images/      # Clean validation dataset from Kaggle
    |    |            ├── Ant/          # Ant images
    |    |            ├── Bee/          # Bee images
    |    |            └── ...           # More images of different insect species organized into folders by class
    │   ├── embs/                     # Contains generated embeddings, labels, and indexes
    │   ├── kairos_output/            #
    │   ├── noisy_images/             # 
    ├── EDA/                          # Preliminary files
        ├── eda.ipynb                 # Exploring the clean Kaggle dataset
         ├── embedding_distribution.ipynb  # Generates plots to see the emebedding overlap between datasets and classes
        └── overlap_asessment.ipynb   # Assessing dataset sizes and overlap
``` -->


```
└── kairos-data-curation/
    ├── data/                                   # Where all data files and generated embeddings are stored
        ├── clean_insect_images/                # Clean validation dataset from Kaggle
            ├── Ant/                            # Ant images
            ├── Bee/                            # Bee images
            └── ...                             # More images of different insect species organized into folders by class
        └── embs/                               # Contains generated embeddings, labels, filepaths, and indexes
    ├── eda/                                    # Preliminary files
        ├── eda.ipynb                           # Exploring the clean Kaggle dataset
        ├── embedding_distribution.ipynb        # Generates plots to see the emebedding overlap between datasets and classes
        └── overlap_asessment.ipynb             # Assessing dataset sizes and overlap
    ├── utils/                                  # Label maps and KAIROS functions
        ├── otdd/                               # Optimal transport dataset distance
            ├── pytorch/                        # Distance functions
            ├── plotting.py                     # Plotting funtions
            └── utils.py                        # Supporting functions
        ├── custom_valuations.py                # KAIROS class
        ├── label_mapping.py                    # Includes iNat species to class mapping in iNat_to_clean_map
        ├── overwrite_package.py                # Overwrites bug in opendataval
        ├── requirements.txt                    # KAIROS dependencies
        └── sample_clean_data.py                # Gets stratified random sample of clean data to create the KAIROS validation set
    ├── src/
        ├── insect_image_noisifier.ipynb        # Adds gaussian noise to insect images
        ├── embedding_extractor.ipynb           # Generates image embeddings for iNat and validation datasets
        ├── kairos_inat_valuation.ipynb         # Uses KAIROS to curate iNat images for fine-tuning ResNet based on clean data.
        ├── resnet_experiments.ipynb            # Experiment for fine-tuning ResNet-50 model via LoRA and partial freezing methods using different fine-tuning datasets               
    └── README.md

```

### D. Project Setup

#### 1. Installation

Create separate environments for different steps to run each script

Steps 1 & 2: Image Noisification & Embedding Extraction (Python 3.11.9)

```bash
# Install dependices for generating CLIP embeddings and initial data processing
python3 -m pip install -r utils/emb_ex_requirements.txt
```

Step 2: Data Valuation (Python 3.9)

```bash
# Install KAIROS dependencies
python3 -m pip install -r utils/kairos_requirements.txt

# Install the benchmark tool opendataval
python3 -m pip install --no-dependencies opendataval

# Fix the data-loading bug in the package by running the overwrite script
python3 utils/overwrite_package.py # Fixes data-loading bugs in opendataval
```

Step 3: ResNet Experiments (Python 3.11)

```bash
# Step 3: Install dependencies for ResNet-50 fine-tuning and evaluation
python3 -m pip install -r utils/experiments_requirements.txt
```

#### 2. Execution Flow

1. insect_image_noisifer.ipynb
2. embedding_extractor.ipynb
3. kairos_inat_valuation.ipynb
4. resnet_experiments.ipynb

##### iii. Model Evaluation (ResNet Experiments):
Use this command to execute the resnet_experiments.ipynb notebook:
```bash
jupyter nbconvert --execute --to html src/resnet_experiments.ipynb

# OR run the command below to also see verbose debug output to trace execution.
jupyter nbconvert --to notebook --execute html src/resnet_experiments.ipynb --debug
```

#### 3. Evaluation & Validation
* Metrics: We utilize Accuracy, F1-Score, AUC, and train time to evaluate model performance.
* Validation: Curation quality is validated by measuring the ....something about kairos plot on poster
* Experiment Tracking: All runs are timestamped and logged in the results/ folders to ensure reproducibility and prevent data loss.

### E. Forward Roadmap
We are actively working to expand the utility of KAIROS-based curation. Our current development focus includes:
* Adapting the MMD-based valuation pipeline for text-based Large Language Models (LLMs) beyond Computer Vision.
* Extending testing to more complex datasets (e.g., medical imaging or satellite data) to further validate model-agnosticism.
* Testing KAIROS performs with increased label noise injection.




