# Data-dreamers-
Text mining and data visualization project

Dataset description
This dataset consists of abstracts and full-text biomedical papers.

Entries
file1.ann: 71697 entries
file2.ann: 45939 entries
File Format
IOB2

Column	Description
id	a string feature.
start	begin character position
end chunk_tags	end character position
ner_tags	a list of classification labels
Labels
Anatomical_system
Cell
Cellular_component
Developing_anatomical_structure
Immaterial_anatomical_entity
Multi-tissue_structure
Organ
Organism_subdivision
Organism_substance
Pathological_formation
Tissue
Example
Ventricular	0	11	B-Multi-tissue_structure
fibrillation	12	24	O


Project track A.Y. 2024-2025 DATA VISUALIZATION & TEXT MINING
The team has to build a text processing pipeline that performs a text classification on the given corpus: all the assigned datasets refer to Entity Extraction use cases, that can be solved applying a text classification approach at token level (Token-based Classification)

The project MUST show:

Data Exploratory Analysis (DEA)

Data preparation, cleaning: to clean the data from the raw dataset provided.
Exploratory Data Analysis using Data Visualization tools to show data variables from statistical distribution (frequency, coverage) to linguistic information (pos, depparse, lemmas)
LDA or NMF can be used, if needed, for studying the text distribution.
Neural Network approach

Use one Neural Network type to classify the data (feed forward, RNN, LSTM, BiLSTM , GRU)
Show metrics for the implementation strategy
Transformer-based Approach

Use a Transformer based / Language Model model to classify the data (*BERT)
Show metrics for the implementation strategy
A comparison about the models

Dashboard

the project MUST implement an interactive DashBoard that combines
the Data Exploratory Analysis with dynamic charts about the dataset
the metrics about the different strategies applied
the ability to have a input box to test the categorizer and to see how it works, moving from a model to another.
Project Artifacts
The project MUST be developed on Jupyter or Colab, and in a customer-ready form that means

well-documented
with descriptions about all the steps
all the materials to reproduce them such as data and models, and instruction to run the dashboard - a Github repository is more than welcome
Datasets
You can find the dataset into your team folder, available in this repository.

Schedule
As all the exams are on Thursdays, the project as to be delivered by previous Tuesday 8pm CET.
