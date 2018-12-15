# SODA
This project aims to automatically identify the Subtype-Oriented Disease Axes (SODA) given the features and multiple labels. It generate the projections of the given features (i.e., the disease axes) that optimally separate the pairwise comparison betweenthe given labels. THe code is included in disease_axis.py.

# Toy Data
We generate some toy data to demonstrate the results of provided the code provided. The toy data is generated via generate_data.py.

The toy data contains 2 files: data.csv and cluster.csv

data.csv contains the patient data. The file is separated with commas. The format of the file is provided as follows:
    
id,     feature_name1,  feature_name2,  feature_name3,     ... # header
XX0001,     -0.94,          -0.14,          -0.91,         ...
XX0002,     0.77,           0.306           0.86           ...
...

cluster.csv contains the patient data. The file is separated with commas. The labels is represeted using integers. "NA" represents that the data is not available. The format of this file is provided as follows:
    
id,         label_name1,    label_name2,    label_name3, ... # header
XX0001,         1,              2,              NA,
XX0002,         0,              2,              0,
...
    

The patients involved in both files are not necessarilly to be the same. The package will automatically match the data and the labels based on the patient id.

# Analyzing the Toy Data

We provide an example that anaylyzes the synthetic data with SODA code:

import disease_axis
M = disease_axis.disease_axis("data.csv", "cluster.csv", savefig = False)


