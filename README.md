# kairos-data-curation
Quarter 2 Project for DSC180AB

The objective of this project is to develop and evaluate a KAIROS-based pipeline for data valuation and curated fine-tuning, applied to an image classification task.

* Our large “messy” dataset will be the **iNaturalist dataset**, containing relevant and irrelevant samples of our target classes (insects): https://github.com/visipedia/inat_comp/tree/master/2021
* Our clean dataset will be the **Kaggle Insects Image dataset** with the desired insect images: https://www.kaggle.com/datasets/ismail703/insects/data


### B. Project Structure
```
└── kairos-data-curation/
    ├── data/                         # where all data files and generated embeddings are stored
        ├── clean_insect_images/      # clean validation dataset from kaggle
            ├── Ant/                  # Ant images
            ├── Bee/                  # Bee images
            └── ...                   # More insect images organized into folders by class
        └── embs/                     # Conatins generated embeddings, labels, filepaths, and ids
    ├── EDA/                          # Preliminary files
        ├── eda.ipynb                 # exploring relevant datasets
        ├── load_cifar100.ipynb       # **Not sure if this is needed**
        └── Overlap_asessment.ipynb   # Assessing dataset sizes and overlap
    ├── utils/                        # Label maps and Kairos functions
        ├── otdd/                     # Optimal transport dataset distance
            ├── pytorch/              # distance functions
            ├── plotting.py           # plotting funtions
            └── utils.py              # supporting functions
        ├── custom_valuations.py      # Kairos class
        ├── overwrite_package.py      # overwrites bug in opendataval
        ├── requirements.txt          # Kairos dependencies
        └── sample_clean_data.py      # Gets stratefied random sample of clean data to create the Kairos validation set
    ├── baseline_resnet50.ipynb       # 
    ├── embedding_extractor.ipynb     # generates image embeddings for iNat and clean data: inat_embs/ and clean_embs/
    ├── kairos_inat_valuation.ipynb   # Uses Kairos to curate iNat images for fine-tuning ResNet based on clean data. Will generate curated_embs.npy
    ├── README.md                     
    ├── resnet50_cifar-100.ipynb      # **Not sure if this is needed**
    ├── resnet50_insects_cv.pth       # 
    └── resnet50_insects.pth          # 
    

```
