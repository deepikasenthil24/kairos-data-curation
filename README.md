# kairos-data-curation
Quarter 2 Project for DSC180AB

The objective of this project is to develop and evaluate a KAIROS-based pipeline for data valuation and curated fine-tuning, applied to an image classification task.

* Our large “messy” dataset will be the **iNaturalist dataset**, containing relevant and irrelevant samples of our target classes (insects): https://github.com/visipedia/inat_comp/tree/master/2021
* Our clean dataset will be the **Kaggle Insects Image dataset** with the desired insect images: https://www.kaggle.com/datasets/ismail703/insects/data


### B. Project Structure
```
└── kairos-data-curation/
    ├── clean_insect_images/          # clean validation dataset from kaggle
        ├── Ant/                      # Ant images
        ├── Bee/                      # Bee images
        └── ...                       # More insect images organized into folders by class
    ├── EDA/                          # Preliminary files
        ├── eda.ipynb                 # exploring relevant datasets
        ├── load_cifar100.ipynb       # **Not sure if this is needed**
        └── Overlap_asessment.ipynb   # Assessing dataset sizes and overlap
    ├── inat_images/                  # Folder of iNat images **should probably remove and just load from huggingface**
    ├── utils/                        # Label maps and Kairos functions
        ├── otdd/                     # Optimal transport dataset distance
            ├── pytorch/              # distance functions
            ├── plotting.py           # plotting funtions
            └── utils.py              # supporting functions
        ├── custom_valuations.py      # Kairos class
        ├── overwrite_package.py      # overwrites bug in opendataval
        └── requirements.txt          # Kairos dependencies
    ├── baseline_resnet50.ipynb       # 
    ├── embedding_extractor.ipynb     # generates image embeddings for iNat and clean data
    ├── kairos_inat_valuation.ipynb   # Uses Kairos to curate iNat images based on clean data
    ├── README.md                     
    ├── resnet50_cifar-100.ipynb      # **Not sure if this is needed**
    ├── resnet50_insects_cv.pth       # 
    ├── resnet50_insects.ipynb        # **Not sure if this is needed if we just use cv**
    └── sample_clean_data.py          # Gets stratefied random sample of clean data 

```

TODO: 
- Add info on what files need to be run to generate required files like embeddings etc.
- Could make sample_clean_data.py adaptable to different sample sizes
