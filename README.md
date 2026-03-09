# Data Valuation for Curated Fine-Tuning using KAIROS
Quarter 2 Project for DSC180AB

**Website:** https://deepikasenthil24.github.io/efficient-fine-tuning-data-curation-kairos/


This repository implements a data curation pipeline using KAIROS to identify high-utility samples from "noisy" datasets, allowing for more efficient model training. We demonstrate this by curating the iNaturalist dataset to improve the fine-tuning of a ResNet-50 model on a target Insects classification task.


### A. Problem Description
Modern machine learning models like **ResNet-50** rely on massive datasets for fine-tuning. However, real-world datasets like **iNaturalist** are often "messy," containing poisoned, mislabeled, or noisy samples that degrade model performance and trustworthiness.

Training on these full datasets is computationally expensive and inefficient. ML engineers currently lack scalable, automated methods to distinguish between high-value samples that drive performance and low-quality data that introduces feature interference. This project addresses the need for a **model-agnostic data valuation** approach, using **KAIROS** to curate high-utility subsets. By identifying the most impactful data points, we maintain model performance while achieving a **93.6% reduction in training time.**


### B. Dataset Context
* Large “messy” dataset: The [**iNaturalist dataset**](https://github.com/visipedia/inat_comp/tree/master/2021), containing relevant samples of our target classes (insect species) as well as significant amounts of irrelevant or "noisy" data (currently integrated via the Hugging Face API for automated ingestion and streaming directly into the pipeline).
* Clean validation dataset: The [**Kaggle Insects Image dataset**](https://www.kaggle.com/datasets/ismail703/insects/data), which contains the desired high-quality insect images used to define our target distribution (clean_insect_images folder currently maintained within the local /data/ directory).


### C. Repository Organization
```
└── kairos-data-curation/
    ├── data/                                   # Where all data files and generated embeddings are stored
        ├── clean_insect_images/                # Clean validation dataset from Kaggle
            ├── Ant/                            # Ant images
            ├── Bee/                            # Bee images
            └── ...                             # More images of different insect species organized into folders by class
        ├── embs/                               # Contains generated embeddings, labels, filepaths, and indexes
        ├── noisy_images/                       # Noisified index mappings (3std, 6std, 9std)
        └── kairos_output/                      # KAIROS valuation and curation indices
    ├── eda/                                    # Preliminary files
        ├── eda.ipynb                           # Explores the clean Kaggle dataset
        ├── embedding_distribution.ipynb        # Generates plots to see the embedding overlap between datasets and classes
        └── overlap_asessment.ipynb             # Assesses dataset sizes and overlap
    ├── kairos_results/                         # Valuation visualizations & metrics
    ├── resnet_results/                         # Experiment logs (LoRA, Unfreezing)
    ├── src/                                    # Core pipeline notebooks
        ├── insect_image_noisifier.ipynb        # Adds Gaussian noise to insect images
        ├── embedding_extractor.ipynb           # Generates image embeddings for iNaturalist and validation datasets
        ├── kairos_inat_valuation.ipynb         # Uses KAIROS to curate iNat images for fine-tuning ResNet based on clean data
        └── resnet_experiments.ipynb            # Experiment for fine-tuning ResNet-50 model via LoRA and partial freezing methods using different fine-tuning datasets (testing notebook)
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
    └── README.md

```

### D. Project Setup

#### 1. Installation

Create separate environments (Python 3.11.9, Python 3.9) for different steps to run each script:

Step 1: Image Noisification & Embedding Extraction (Python 3.11.9)

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
Run the following notebooks:
1. insect_image_noisifer.ipynb
* Generates a corrupted version of the iNaturalist dataset by injecting Gaussian noise at controlled intensities ($30\sigma, 60\sigma, 90\sigma$). It outputs a set of labeled noisy image paths or augmented image tensors used to stress-test the curation's resilience.
  
2. embedding_extractor.ipynb
* Processes both the clean Kaggle and noisy iNaturalist images through a CLIP-ViT-L-14 backbone. It saves high-dimensional feature vectors as .npy files in the data/embs/ directory, which serve as the mathematical input for the valuation algorithm.

3. kairos_inat_valuation.ipynb
* Performs the MMD-based distribution comparison to rank every iNaturalist sample. Outputs .npy files ranked indices, effectively identifying the high-utility samples within the noisy set.
* If cell 5 kills the kernel due to RAM limtations, try reducing the INAT_SUB_SIZE

4. resnet_experiments.ipynb (optional: terminal command documented below)


##### iii. Model Evaluation (ResNet Experiments):
Use this command to execute the resnet_experiments.ipynb notebook:
```bash
jupyter nbconvert --execute --to html src/resnet_experiments.ipynb

# OR run the command below to also see verbose debug output to trace execution.
jupyter nbconvert --to notebook --execute html src/resnet_experiments.ipynb --debug
```
The resnet_experiments.ipynb notebook logs results directly to notebook cells and archives all artifacts within the resnet_results/ directory. For organized tracking, each run generates a dedicated folder named by the Experiment Name, with a sub-folder containing the execution timestamp. The following outputs are generated: Performance Metrics (captured logs for accuracy, AUC, weighted F1 scores, and both training and test times) and Performance Plots (Confusion matrices and scaling plots for Metric vs. Data Size)


#### 3. Evaluation & Validation
* Metrics: We utilize Accuracy, F1-Score, AUC, and train time to evaluate model performance.
* Experiment Tracking: All runs are timestamped and logged in the results/ folders to ensure reproducibility and prevent data loss.

### E. Forward Roadmap
We are actively working to expand the utility of KAIROS-based curation. Our current development focus includes:
* Adapting the MMD-based valuation pipeline for text-based Large Language Models (LLMs) beyond Computer Vision.
* Extending testing to more complex datasets (e.g., medical imaging or satellite data) to further validate model-agnosticism.
* Testing KAIROS performs with increased label noise injection.




