# Wildfire Prediction Model

This project builds a machine learning model to predict wildfire occurrences using the modified Next Day Wildfire Spread dataset.
video explanation = https://drive.google.com/drive/folders/1QRzTXa_GYou7waC_rEkxHVY6ySUm-j-y?usp=sharing

## Project Organization

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. The dataset consists of TFRecord files containing geospatial and meteorological features for wildfire prediction.

3. Start with the data exploration notebook to understand the dataset structure.

## Dataset

The project uses the modified Next Day Wildfire Spread dataset in TFRecord format, containing:
- Geospatial raster data
- Meteorological features
- Historical wildfire occurrence data
- Temporal features

## License

This project is licensed under the MIT License.
