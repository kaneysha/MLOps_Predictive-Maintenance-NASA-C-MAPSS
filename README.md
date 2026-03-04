# MLOps_Predictive-Maintanance-NASA-C-MAPSS

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Predictive Maintanance untuk Mesin Industri Berbasis Continual Learning Menggunakan NASA C-MAPSS

## Tujuan
- Memprediksi waktu kegagalan mesin sebelum benar-benar terjadi sehingga tindakan pemeliharaan dapat dilakukan secara proaktif
- Mengoptimalkan jadwal perawatan
- Mengurangi downtime tidak terencana
- Menekan biaya operasional

## Cara Menjalankan Codespaces
1. Buka repository Github https://github.com/kaneysha/MLOps_Predictive-Maintenance-NASA-C-MAPSS
2. Klik tombol 'Code'
3. Klik tab 'Codespaces'
4. Klik "Create codespace on main"
5. Tunggu hingga proses build selesai

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mlops_predictive_maintanance_nasa_c_mapss and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mlops_predictive_maintanance_nasa_c_mapss   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mlops_predictive_maintanance_nasa_c_mapss a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

