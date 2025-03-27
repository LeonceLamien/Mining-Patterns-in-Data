#!/usr/bin/env python3

import sys
import heapq

class Spade:

    def __init__(self, pos_filepath, neg_filepath, k):
        self.pos_transactions = get_transactions(pos_filepath)
        self.neg_transactions = get_transactions(neg_filepath)
        self.k = k
        
    def min_top_k(self):
        """
        Mine the top-k sequential patterns using the SPADE algorithm
        """
        # Convert transactions to SPADE vertical representation
        positive_spade = spade_repr_from_transaction(self.pos_transactions)
        negative_spade = spade_repr_from_transaction(self.neg_transactions)
    
        # Get all unique items
        all_unique_items = sorted(list(set(positive_spade['repr'].keys()).union(negative_spade['repr'].keys())))
    
        # Initialize the list of top-k patterns
        top_k_patterns = []
    
        # Handle single-item patterns
        for item in all_unique_items:
            positive_support = len(positive_spade['covers'][item])
            negative_support = len(negative_spade['covers'][item])
            total_support = positive_support + negative_support
        
            pattern = [item]
        
            if len(top_k_patterns) < self.k:
                heapq.heappush(top_k_patterns, (total_support, pattern, positive_support, negative_support))
            elif total_support > top_k_patterns[0][0]:
                heapq.heapreplace(top_k_patterns, (total_support, pattern, positive_support, negative_support))
                
        # Perform depth-first search to find longer patterns
        self.depth_first_search(positive_spade, negative_spade, all_unique_items, top_k_patterns)
        
        # Sort by total support in descending order
        result = sorted(top_k_patterns, key=lambda x: (-x[0], x[1]))
        
        # Print the top-k patterns
        for total_support, pattern, pos_support, neg_support in result:
            pattern_str = '[' + ', '.join(pattern) + ']'
            print(f"{pattern_str} {pos_support} {neg_support} {total_support}")
    
        return result
    
    
    def depth_first_search(self, positive_spade, negative_spade, items, top_k_patterns):
        """
        Recursive depth-first search to find frequent sequences
        """
        # Create a stack for depth first search
        stack = []
    
        # Initialize with single-item patterns
        for item in items:
            stack.append(([item], positive_spade['repr'][item] if item in positive_spade['repr'] else [], 
                          negative_spade['repr'][item] if item in negative_spade['repr'] else []))
    
        # Perform depth-first search
        while stack:
            pattern, pos_occurrences, neg_occurrences = stack.pop()  
            
            # Try extending the pattern with each item
            for item in items:
                new_pattern = pattern + [item]
            
                # Calculate new occurrences and supports
                new_pos_occurrences = self.get_new_occurrences(pos_occurrences, positive_spade['repr'][item] if item in positive_spade['repr'] else [])
                new_neg_occurrences = self.get_new_occurrences(neg_occurrences, negative_spade['repr'][item] if item in negative_spade['repr'] else [])
            
                pos_support = len(set(tid for tid, _ in new_pos_occurrences))
                neg_support = len(set(tid for tid, _ in new_neg_occurrences))
                total_support = pos_support + neg_support                       
                
                # Update top-k patterns
                if len(top_k_patterns) < self.k:
                    heapq.heappush(top_k_patterns, (total_support, new_pattern, pos_support, neg_support))
                    min_support = top_k_patterns[0][0] if top_k_patterns else 0
                elif total_support > top_k_patterns[0][0]:
                    heapq.heapreplace(top_k_patterns, (total_support, new_pattern, pos_support, neg_support))
                    min_support = top_k_patterns[0][0]
                else:
                    min_support = top_k_patterns[0][0]
            
                # If new pattern has enough support, add it to the stack for further extension
                if total_support >= min_support and (new_pos_occurrences or new_neg_occurrences):
                    stack.append((new_pattern, new_pos_occurrences, new_neg_occurrences))       
                
                
    def get_support(self, pattern, spade_repr):
        """
        Calculate support of a pattern in a dataset
        """
        if not pattern:
            return 0
            
        # Return supports is single pattern
        if len(pattern) == 1:
            return len(set(tid for tid, _ in spade_repr.get(pattern[0], [])))
        
        # Check the occurrences for longer patterns
        occurrences = spade_repr.get(pattern[0], [])
        for i in range(1, len(pattern)):
            occurrences = self.get_new_occurrences(occurrences, spade_repr.get(pattern[i], []))
        
        return len(set(tid for tid, _ in occurrences))

    def get_new_occurrences(self, prefix_occurrences, item_occurrences):
        """
        Finds occurrences of the extended pattern by comparing positions
        """
        if not prefix_occurrences or not item_occurrences:
            return []
        
        new_occurrences = []
        
        # Group occurrences by transaction id
        prefix_dict = {}
        for tid, pos in prefix_occurrences:
            if tid not in prefix_dict:
                prefix_dict[tid] = []
            prefix_dict[tid].append(pos)
        
        # For each item occurrence, check if it can extend the prefix
        for tid, pos in item_occurrences:
            if tid in prefix_dict:
                prefix_positions = prefix_dict[tid]
                for prefix_pos in prefix_positions:
                    if pos > prefix_pos:  # Item occurs after the prefix
                        new_occurrences.append((tid, pos))
                        break
        
        return new_occurrences

    
    
    def get_feature_matrices(self):
        return {
            'train_matrix': [],
            'test_matrix': [],
            'train_labels': [],
            'test_labels': [],
        }

    def cross_validation(self, nfolds):
        pos_fold_size = len(self.pos_transactions) // nfolds
        neg_fold_size = len(self.neg_transactions) // nfolds
        for fold in range(nfolds):
            print('fold {}'.format(fold + 1))
            pos_train_set = {i for i in range(len(self.pos_transactions)) if i < fold*pos_fold_size or i >= (fold+1)*pos_fold_size}
            neg_train_set = {i for i in range(len(self.neg_transactions)) if i < fold*neg_fold_size or i >= (fold+1)*neg_fold_size}

            self.mine_top_k()
            
            m = self.get_feature_matrices()
            classifier = tree.DecisionTreeClassifier(random_state=1)
            classifier.fit(m['train_matrix'], m['train_labels'])

            predicted = classifier.predict(m['test_matrix'])
            accuracy = metrics.accuracy_score(m['test_labels'], predicted)
            print(f'Accuracy: {accuracy}')

def get_transactions(filepath):
    transactions = []
    with open(filepath) as f:
        new_transaction = True
        for line in f:
            if line.strip():
                if new_transaction:
                    transactions.append([])
                    new_transaction = False
                element = line.split(" ")
                assert(int(element[1]) - 1 == len(transactions[-1]))
                transactions[-1].append(element[0])
            else:
                new_transaction = True
    return transactions

def spade_repr_from_transaction(transactions):
    spade_repr = {}
    covers = {}
    for tid, transaction in enumerate(transactions):
        for i, item in enumerate(transaction):
            try:
                covers[item].add(tid)
            except KeyError:
                covers[item] = {tid}
            try:
                spade_repr[item].append((tid, i))
            except KeyError:
                spade_repr[item] = [(tid, i)]
    return {'repr': spade_repr, 'covers': covers}


def main():
    s = Spade(pos_filepath, neg_filepath, k)
    result = s.min_top_k()
    return result

if __name__ == '__main__':
    pos_filepath = sys.argv[1]
    neg_filepath = sys.argv[2]
    k = int(sys.argv[3])
    main()