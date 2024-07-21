# SHARQ

this repository contains the code for calculating **SHARQ** - an algorithm for quantifying an element’s contribution to a set of association rules based on Shapley values.

# Source Code
The source code is located in the `src` directory. 
Under this directory, there is the code for:
1. Mining association rules from datasets according to all configurations mentioned in the paper. The generated rules set are saved in the `Rules` directory.
2. Calculating the SHARQ values for each element in the dataset. The calculated returned as a dictionary where the key is the element and the value is the SHARQ value. There is also an option to calculate the Influence, I_top and SHARQ approximation values.
3. Visualizing the SHARQ values of all elements appear in the rules set.
4. Calculating the normalized SHARQ values for each element in the dataset. The values returned as a dictionary where the key is the element and the value is the normalized SHARQ value.

# Experiments Datasets
The datasets used in the experiments are located in the `Datasets` directory.
We used 4 datasets in the experiments:
1. 'Adults' - an individual’s annual income results from various factors. Containing 49K rows and 16 columns.
2. 'Spotify songs' -  a dataset of Spotify tracks. Each track has some audio features associated with it. Containing 174K rows and 22 columns
3. 'Flight Delays' - This database contains scheduled and actual departure and arrival times, reason of delay etc. Containing 5.8M rows and 30 columns
4. 'Isolet' - This dataset generated from the recordings of spoken letters from the English alphabet. Containing 7.8K rows and 617 columns

# Use Cases and Examples
An example of a simple use case is provided in the `example_notebook.ipynb` file. In this notebook we generate an association rules set from the 'Adults' dataset and calculate the SHARQ values for each element in the dataset.
We than visualize the SHARQ values of the top scored, bottom scored and frequent elements in the dataset and calculate the normalized SHARQ values for each element in the dataset. 
There is an option to generate the 'Adults' rules set from the notebook or used an already generated one from the `Rules` directory.

# More
If you have any questions or need further assistance, please contact us.



