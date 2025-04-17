#!/usr/bin/env python3

import sys
import numpy as np
from sklearn import tree
from sklearn.model_selection import KFold
from collections import Counter
import supervised_miner

def transform_to_feature_set(transactions, patterns):
    """
    Transform transactions into a feature matrix based on pattern presence
    Each row represents a transaction, each column represents a pattern
    """
    feature_matrix = []
    
    for transaction in transactions:
        # For each transaction, create a feature vector of pattern presence
        features = []
        for pattern in patterns:
            # Check if pattern is a subsequence of transaction
            if is_subsequence(pattern, transaction):
                features.append(1)
            else:
                features.append(0)
        feature_matrix.append(features)
    
    return np.array(feature_matrix)

def is_subsequence(pattern, transaction):
    """
    Check if pattern is a subsequence of transaction
    """
    if not pattern:
        return True
    
    # Find the pattern using a greedy approach
    i = 0  # pattern index
    for item in transaction:
        if item == pattern[i]:
            i += 1
            if i == len(pattern):
                return True
    
    return False

def train_and_evaluate(pos_filepath, neg_filepath, k, min_freq, num_folds, verbose):
    """
    Train a decision tree classifier using top-k patterns according to WRAcc and evaluate with cross-validation
    """
    # Load transactions
    pos_transactions = supervised_miner.get_transactions(pos_filepath)
    neg_transactions = supervised_miner.get_transactions(neg_filepath)
    
    # Combine transactions and create labels
    all_transactions = pos_transactions + neg_transactions
    y = np.array([1] * len(pos_transactions) + [-1] * len(neg_transactions))
    
    # Setup k-fold cross validation
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=1)
    
    total_accuracy = 0
    fold_num = 1
    
    # For producing a contingency table at the end
    all_predictions = []
    all_true_labels = []
    
    for train_index, test_index in kf.split(all_transactions):
        if verbose:
            print(f"fold {fold_num}")
        
        # Split data into train and test sets
        train_transactions = [all_transactions[i] for i in train_index]
        test_transactions = [all_transactions[i] for i in test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        
        # Split train transactions into positive and negative
        pos_train_indices = [i for i, label in enumerate(y_train) if label == 1]
        neg_train_indices = [i for i, label in enumerate(y_train) if label == -1]
        
        pos_train = [train_transactions[i] for i in pos_train_indices]
        neg_train = [train_transactions[i] for i in neg_train_indices]
        
        # Find top-k patterns using the supervised SPADE algorithm with WRAcc
        spade = supervised_miner.Spade(pos_filepath, neg_filepath, k)
        spade.pos_transactions = pos_train
        spade.neg_transactions = neg_train
        
        patterns_with_scores = spade.min_top_k_wracc()
        
        # Filter patterns by minimum frequency
        filtered_patterns = []
        for pattern, pos_support, neg_support, wracc_score in patterns_with_scores:
            # Calculate overall frequency
            total_support = pos_support + neg_support
            total_train_size = len(pos_train) + len(neg_train)
            frequency = total_support / total_train_size
            
            if frequency >= min_freq:
                filtered_patterns.append(pattern)
                # Print pattern details if verbose is enabled
                if verbose:
                    pattern_str = '[' + ', '.join(pattern) + ']'
                    print(f"{pattern_str} {pos_support} {neg_support} {wracc_score}")
        
        if not filtered_patterns:
            if verbose:
                print("No patterns meet the minimum frequency threshold")
            fold_num += 1
            continue
        
        # Transform train and test data into feature matrices
        X_train = transform_to_feature_set(train_transactions, filtered_patterns)
        X_test = transform_to_feature_set(test_transactions, filtered_patterns)
        
        # Train decision tree classifier
        classifier = tree.DecisionTreeClassifier(random_state=1)
        classifier.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = classifier.predict(X_test)
        
        # Display predictions if verbose is enabled
        if verbose:
            print(' '.join(map(str, y_pred.tolist())))
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)
        if verbose:
            print(f"accuracy: {accuracy:.5f}")
        total_accuracy += accuracy
        
        # Store predictions and true labels for contingency table
        all_predictions.extend(y_pred)
        all_true_labels.extend(y_test)
        
        fold_num += 1
        if verbose:
            print()  # Blank line between folds
    
    # Calculate and display mean accuracy
    mean_accuracy = total_accuracy / num_folds
    if verbose:
        print(f"Mean Accuracy: {mean_accuracy:.5f}")
    
    # Display contingency table
    conf_matrix = contingency_table(all_true_labels, all_predictions)
    if verbose:
        print("\nContingency Table:")
        print(f"True Positive: {conf_matrix['TP']}")
        print(f"False Positive: {conf_matrix['FP']}")
        print(f"True Negative: {conf_matrix['TN']}")
        print(f"False Negative: {conf_matrix['FN']}")
    
    print(f"Mean Accuracy: {mean_accuracy:.5f}")
    return mean_accuracy

def contingency_table(y_true, y_pred):
    """
    Create a contingency table for binary classification
    """
    result = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            result['TP'] += 1
        elif true == -1 and pred == 1:
            result['FP'] += 1
        elif true == -1 and pred == -1:
            result['TN'] += 1
        elif true == 1 and pred == -1:
            result['FN'] += 1
    
    return result

def main():
    pos_filepath = sys.argv[1]
    neg_filepath = sys.argv[2]
    k = int(sys.argv[3])
    min_freq = float(sys.argv[4])
    num_folds = int(sys.argv[5])
    
    train_and_evaluate(pos_filepath, neg_filepath, k, min_freq, num_folds, True)

if __name__ == '__main__':
    main()


###################################################################################################################
#### NOTES ########################################################################################################
#### In order to have a clear output, we should comment the print statements in the supervised_miner.py  ##########
#### file particularly in the min_top_k_wracc function  ###########################################################
###################################################################################################################