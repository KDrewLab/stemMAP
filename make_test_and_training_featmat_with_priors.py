##Make test and training featmat with priors
##authors: arajar4
##Date: 2025-06-25

'''
This code takes in a feature matrix with correlation scores derived from context-specific CF-MS data and 
prior-features from a larger generic database,
test and training pairs for context-specific dataset, 
test and training pairs from priors dataset.
- Checks for overlap and removes the cross-overlap between the two datasets' test and train set.
- Outputs feature matrix for all categories of protein pairs
'''

#Import necessary modules
from __future__ import print_function
import argparse
import pandas as pd
import numpy as np
import os.path
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from matplotlib_venn import venn3
import random
import scipy
from sklearn import metrics
from sklearn.metrics import precision_recall_curve

def build_test_train_featmat():
    parser = argparse.ArgumentParser(description="Builds test and training feature matrix with priors and labels (format: ID1 ID2 ,feat1, prior_feat1, ..., label")

    parser.add_argument("--featmat_with_priors", action="store", dest="featmat_priors", required=True,
                                    help="Filename of the feature matrix with raw and priors features,
                                    ID colum labelled 'ID' pairs separated by space and a csv file are required")

    parser.add_argument("--train_pos_pairs", action="store", dest="train_pos_pairs", required=True,
                                    help="File of protein pairs, one per line")

    parser.add_argument("--train_neg_pairs", action="store", dest="train_neg_pairs", required=True,
                                    help="File of protein pairs, one per line")

    parser.add_argument("--test_pos_pairs", action="store", dest="test_pos_pairs", required=True,
                                    help="File of protein pairs, one per line")

    parser.add_argument("--test_neg_pairs", action="store", dest="test_neg_pairs", required=True,
                                    help="File of protein pairs, one per line")
    
    parser.add_argument("--prior_train_pos_pairs", action="store", nargs='?', dest="prior_train_pos_pairs", required=False,
                                    help="File of protein pairs, one per line")
    
    parser.add_argument("--prior_train_neg_pairs", action="store", nargs='?', dest="prior_train_neg_pairs", required=False,
                                    help="File of protein pairs, one per line")

    parser.add_argument("--prior_test_pos_pairs", action="store", nargs='?', dest="prior_test_pos_pairs", required=False,
                                    help="File of protein pairs, one per line")

    parser.add_argument("--prior_test_neg_pairs", action="store", nargs='?', dest="prior_test_neg_pairs", required=False,
                                    help="File of protein pairs, one per line")

    parser.add_argument("--context_specific_proteins", action="store", dest="context_sp_proteins", required=False,
                                    help="A file containing list of proteins only expressed in the specific context")

    parser.add_argument("--output_train_featmat", action="store", dest="train_featmat", required=True,
                                    help="Output training featmat file name")

    parser.add_argument("--output_test_featmat", action="store", dest="test_featmat", required=True,
                                    help="Output test featmat file name")

    parser.add_argument("--sep", action="store", dest="sep", required=True,
                                    help="Separator for input files")
    
    args = parser.parse_args()

    featmat_priors = pd.read_csv(args.featmat_priors, sep = args.sep)
    print(f"Input featmat has {featmat.shape[0]} rows")
    print(featmat_priors.columns())

    ## Make frozenset indices out of the ID columns
    featmat_priors["Frozen_Set"] = [frozenset((x.split())) for x in featmat_priors["ID"]]
    featmat_priors = featmat_priors.set_index("Frozen_Set", drop = True)
    
    print(featmat_priors.head())

    ##Function to make a list of frozen set pairs from the text files
    def read_protein_pairs(set_name,file_path):
        protein_pairs = set()
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()  # splits on any whitespace
                if len(parts) != 2:
                    continue  # skip malformed lines
                pair = frozenset(parts) # make frozen set out of the pairs
                protein_pairs.add(pair)
        print(f"Number of {set_name}:{len(protein_pairs)}")
        return protein_pairs

    train_pos = read_protein_pairs("training positives",args.train_pos_pairs)
    train_neg = read_protein_pairs("training negatives",args.train_neg_pairs)
    test_pos = read_protein_pairs("test positives",args.test_pos_pairs)
    test_neg = read_protein_pairs("test negatives",args.test_neg_pairs)


    if args.prior_train_pos_pairs:
        prior_train_pos = read_protein_pairs("Prior training positives",args.prior_train_pos_pairs)
        prior_train_neg = read_protein_pairs("Prior training negatives",args.prior_train_neg_pairs)
        prior_test_pos = read_protein_pairs("Prior test positives",args.prior_test_pos_pairs)
        prior_test_neg = read_protein_pairs("Prior test negatives",args.prior_test_neg_pairs)
        
        ##removing cross overlap means
        ##eg. train_pos - (prior_test_pos | prior_train_neg| prior_test_neg)
        
        def clean_benchmark(working_set, *overlaps):
            cleaned_set = working_set - set().union(*overlaps)
            print("Length of the working set:",len(cleaned_set))
            
            #cleaned_list = [tuple(sorted(fs)) for fs in cleaned_set]
            #cleaned_df = pd.DataFrame(cleaned_list, columns = ["ID1","ID2"])
            
            return cleaned_set

        ##removing cross overlap
        train_pos_cleaned = clean_benchmark(train_pos, prior_train_neg, prior_test_pos, prior_test_neg)
        train_neg_cleaned = clean_benchmark(train_neg, prior_train_pos, prior_test_pos, prior_test_neg)

        test_pos_cleaned = clean_benchmark(test_pos, prior_train_pos, prior_train_neg, prior_test_neg)
        test_neg_cleaned = clean_benchmark(test_neg, prior_train_pos, prior_train_neg, prior_test_pos)

        ##If you have a context specific proteome, we add training/test positive pairs if atleast one of the proteins in the prior_benchmark is context specific

        if args.context_sp_proteins:

            cs_proteins = pd.read_csv(args.context_sp_proteins)
            cs_proteins_set = set(cs_proteins[0])
            print(len(cs_proteins_set))

            prior_benchmark_list = [prior_train_pos, prior_train_neg, prior_test_pos, prior_test_neg]
            
            def add_context_specific_pairs(cs_proteins_set):
                for key in prior_benchmark_list:
                    cs_subset = {pair for pair in key if pair & cs_protein_set}
                    key = key.union(cs_subset)

    ##Build a feature matrix, add labels and write the output file
    
    def filter_featmat(working_set,featmat_priors)
        cleaned_df = featmat_priors[featmat_priors.index.isin(working_set)]
                    
        print(f"No of protein pairs = {len(working_set)}")
        print(f"No of protein pairs with features = {cleaned_df.shape[0]}")
                    
        combined_pairs = pd.DataFrame([tuple(sorted(fs)) for fs in cleaned_df.index])

        return cleaned_df, combined_pairs
    
    train_pos_cleaned, train_pos_pairs =filter_featmat(train_pos_cleaned, featmat_priors)
    train_neg_cleaned, train_neg_pairs =filter_featmat(train_neg_cleaned, featmat_priors)
    test_pos_cleaned, test_pos_pairs =filter_featmat(test_pos_cleaned, featmat_priors)
    test_neg_cleaned, test_neg_pairs =filter_featmat(test_neg_cleaned, featmat_priors)


   



