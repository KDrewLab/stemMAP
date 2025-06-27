##Add priors to feature matrix
##authors: arajar4
##Date: 2025-06-25

'''
This code takes in a feature matrix with correlation scores derived from context-specific CF-MS data,
predictions file from a larger generic database as 'priors'.
- Checks for duplicate indices
- Outputs a general feature matrix with prior-multiplied features columns.
'''

#Import necessary modules
from __future__ import print_function
import argparse

import pandas as pd
import numpy as np
import os.path
import seaborn as sns
import matplotlib.pyplot as plt
import time
import itertools


def build_featmat_with_priors():
    parser = argparse.ArgumentParser(description="""Builds test and training feature matrix with priors,
                                                    format: ID1, ID2, feat1, feat2, ...,prior_feat1, prior_feat2...
                                                    **All input files need to have headers and same delimiters**""")
    parser.add_argument("--input_featmat", action="store", dest="featmat", required=False,
                                    help="""Filename of the feature matrix with features, csv or tsv file""")

    parser.add_argument("--prior_predictions", action="store", dest='prior_predictions', required=True,
                                    help="""protein pairs and their confidence of interaction in a tsv file. 
                                    Note that the protein pairs must be mapped to the target organism before proceeding""")
    
    parser.add_argument("--replace_priors", action = "store_true", dest = "replace_priors", required = False,
                                    help="""If you want all the non exisitent prior data to be replaced by random small numbers
                                    between 0.000001 and 0.000009""")
    
    parser.add_argument("--output_file", action="store", dest="out_filename", required=True,
                                    help="Output featmat file with prior columns")
    
    parser.add_argument("--sep", action="store", dest="sep", required=True, 
                                    help="Separator for input files")
  
    args = parser.parse_args()


    ##Read in featmat into df
    featmat = pd.read_csv(args.featmat, sep = args.sep)
    print(f"Input featmat: {featmat.shape[0]} rows")


    ##Make frozenset indices out of the ID columns
    featmat["Frozen_set"] = featmat.apply(lambda row: frozenset((row[0], row[1])), axis =1)
    featmat = featmat.set_index("Frozen_set", drop = True)
    print(featmat.head())

    ##Check for duplicated indices
    if featmat.index.duplicated().sum() > 0:
        print(f"Error!: There are {featmat.index.duplicated().sum()} duplicated indices in featmat. Go to https://github.com/marcottelab/leca-proteomics/blob/main/scripts/build_featmat.py") 
    
    print("\n")

    ##Read in the prior predictions files
    prior_predictions = pd.read_csv(args.prior_predictions, sep = args.sep)
    ##Make frozen set indices
    prior_predictions["Frozen_set"] = prior_predictions.apply(lambda row: frozenset((row[0], row[1])), axis =1)
    prior_predictions = prior_predictions.set_index("Frozen_set", drop = True)

    ##rename columns
    prior_predictions.columns = ["ID1","ID2","prior_prediction"]
    print(f"Prior predictions: {prior_predictions.shape[0]} rows")
    print(prior_predictions.head())
    prior_predictions["prior_prediction"] = prior_predictions["prior_prediction"].astype(float)

    ##Merge priors with the pandas dataframe
    featmat_priors = pd.merge(prior_predictions[["prior_prediction"]], featmat, how = "right", left_index = True,right_index = True) ##Right merge because I want the prior column in the beginning.

    print(f"Rows in featmat before merging priors :{featmat.shape[0]} should be equal to the rows after merge:{featmat_priors.shape[0]}")

    ##Nan values are filled to zero
    featmat_priors = featmat_priors.fillna(0)
    print(featmat_priors.head())

    feature_cols = featmat_priors.columns[3:] #define columns that are raw features
    print("Columns to be calculated priors for:\n",feature_cols)

    ##Create a df of priors (.multiply is more efficient)
    prior_features = featmat_priors[feature_cols].multiply(featmat_priors['prior_prediction'], axis=0)
    prior_features.columns = [f"Prior_{col}" for col in prior_features.columns] # rename columns with priors
   
    if args.replace_priors:
        prior_features = prior_features.applymap(lambda x: np.random.uniform(0.000001, 0.000009) if x == 0 else x)
    
    featmat_priors = pd.concat([featmat_priors, prior_features], axis = 1) #concat dataframes

    ## Pop the 'ID' columns
    #Insert it at the beginning (index 0)
    cols = ['ID1', 'ID2'] + [col for col in featmat_priors.columns if col not in ['ID1', 'ID2']]
    featmat_priors = featmat_priors[cols]

    print("Final_prior_added_featmat:")
    print(featmat_priors)

    ##Write to csv
    featmat_priors.to_csv(args.out_filename, sep = args.sep, index = False)

if __name__ == "__main__":
    build_featmat_with_priors()





