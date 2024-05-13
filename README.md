# AMAR: Ask Me Anything with RAG

## Table of Contents

## Objective

The objective of this project is to create a simple and easy to use tool that allows users to ask questions about local data. It works by serving local LLM models and a RAG model to answer questions about the data.

## Repository Structure

The repository is structured as follows:


```

├── README.md <- The top-level README for developers using this project.
|
|
├── data <- folder where the data is stored, the test is already saved
│
├── jobs <- part of model development, adjustment of model hyperparameters
│
├── lib <- folder where the model weights are stored
│
├── notebooks <- folder where jupyter notebooks are located
|   ├── data_processing <- Folder for dataset preprocessing     
|   ├── eda <- Data Analysis
|   ├── ocr <- OCR analysis and comparison 
|   ├── test <- Checking the functionality of each stage
|   ├── train_icons.ipynb <- Train stage Icon Classification
|   └── train_yolo.ipynb <- Train stage YOLO Icon Detection
│
├── scripts  <- folder that contains the scripts used in the PoC
|   ├── inference.py <- code used for inference        
|   ├── label_studio_to_yolo.py <- development code, converting the annotated data from label studio to yolo structure
|   ├── partitioning_random.py <- development code, used for partitioning purposes
|   └── train.py <- development code, used for training purposes
│
├── CHANGELOG.md  <- changelog registered every week of the changes applied in the PoC
│
├── Dockerfile <- Docker for the deployment of the web app
|
├── VIEWS.md  <- View structures used in Label Studio for the annotation process
│
├── requirements.txt <- List of the libraries used for Inference (Pyenv)
|
├── conda-environment.yaml <- List of the libraries used for Inference (Conda)
|
├── dev_requirements.txt <- List of the libraries used for Notebooks (Pyenv)
|
├── dev-conda-environment.yaml <- List of the libraries used for Notebooks (Conda)
|
└── app.py <- code to deploy a web app with Streamlit 

```

## Virtual Environment

To ensure the reproducibility of the project, it is recommended to create a virtual environment.
[Conda](https://www.anaconda.com) is a package manager that allows you to create virtual environments with different Python versions and install the required dependencies.

The following commands can be used to create a virtual environment using conda:

```bash
conda env create -f dev-environment.yaml
```

### Tools used

The following tools were used in the development of the project:
  - [SuryaOCR](https://github.com/VikParuchuri/surya): OCR tool used to extract text from scanned documents.
  - [Ollama](https://github.com/ollama/ollama): Get up and running with large language models locally. Used for embedding text data and LLMs serving.
  - [ChromaDB](https://www.trychroma.com): Embedding database used to store and retrieve embeddings.
  - [Streamlit](https://streamlit.io): Streamlit turns data scripts into shareable web apps in minutes. All in pure Python. No front‑end experience required.


## Running the Web App

```bash
streamlit run app.py
```
