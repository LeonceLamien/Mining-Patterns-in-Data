#!/usr/bin/env python3

import sys
import time

class Spade:

    def __init__(self, pos_filepath, neg_filepath, k):
        self.pos_transactions = get_transactions(pos_filepath)
        self.neg_transactions = get_transactions(neg_filepath)
        self.k = k
        
    def min_top_k(self):
        """
        Mine the top-k sequential patterns using the SPADE algorithm
        """

        positive_spade = spade_repr_from_transaction(self.pos_transactions)
        negative_spade = spade_repr_from_transaction(self.neg_transactions)
    
        all_unique_items = sorted(list(set(positive_spade['repr'].keys()).union(negative_spade['repr'].keys())))
    
        all_patterns = []
        to_sup=set()
        for item in all_unique_items:
            positive_support = len(positive_spade['covers'][item])
            negative_support = len(negative_spade['covers'][item])
            total_support = positive_support + negative_support
            to_sup.add(total_support)
            pattern = [item]
            all_patterns.append((total_support, pattern, positive_support, negative_support))
        
        # Find longer patterns using DFS
        self.depth_first_search(positive_spade, negative_spade, all_unique_items,to_sup, all_patterns)
        
        # Sort patterns by total support then by pattern lexicographically
        all_patterns.sort(key=lambda x: (-x[0], x[1]))
        
        # # Select top-k patterns
        # if len(all_patterns) <= self.k:
        #     result = all_patterns
        # else:
        #     # Get the k-th pattern's support
        #     k_th_support = all_patterns[self.k-1][0]
            
        #     # Find how many patterns have the same support as the k-th pattern
        #     same_support_count = sum(1 for p in all_patterns if p[0] == k_th_support)

        #     # Find the index of the first pattern with this support
        #     first_index = next(i for i, p in enumerate(all_patterns) if p[0] == k_th_support)
            
        #     # Include all patterns with support greater than k-th support, plus all patterns with exactly the k-th support
        #     result = all_patterns[:first_index + same_support_count]
        
        # Print the result
        to_sup=sorted(list(to_sup),reverse=True)[:5]
        for total_support, pattern, pos_support, neg_support in all_patterns:
            if total_support in to_sup:
                pattern_str = '[' + ', '.join(pattern) + ']'
                print(f"{pattern_str} {pos_support} {neg_support} {total_support}")
            
        return all_patterns

    def depth_first_search(self, positive_spade, negative_spade, items,to_sup, all_patterns):
        """
        Depth-first search to find all sequential patterns
        """
        # Stack for DFS
        stack = []
        
        # Initialize the stack with single items
        for item in sorted(items):
            stack.append(([item], 
                         positive_spade['repr'].get(item, []), 
                         negative_spade['repr'].get(item, [])))
        
        pattern_count = 0
        # Run DFS
        while stack:
            pattern, pos_occurrences, neg_occurrences = stack.pop()
            
            # Try extending the pattern with each possible item
            for item in sorted(items, reverse=True):
                new_pattern = pattern + [item]
                
                # Get new occurrences
                new_pos_occurrences = self.get_new_occurrences(pos_occurrences, positive_spade['repr'].get(item, []))
                new_neg_occurrences = self.get_new_occurrences(neg_occurrences, negative_spade['repr'].get(item, []))
                
                # Only consider patterns with at least one occurrence
                if not new_pos_occurrences and not new_neg_occurrences:
                    continue
                
                # Calculate supports
                pos_support = len(set(tid for tid, _ in new_pos_occurrences))
                neg_support = len(set(tid for tid, _ in new_neg_occurrences))
                total_support = pos_support + neg_support
                to_sup.add(total_support)
                # Add the pattern to our results
                all_patterns.append((total_support, new_pattern, pos_support, neg_support))
                pattern_count += 1
                
                # Push to stack for further extension
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
        
        prefix_dict = {}
        for tid, pos in prefix_occurrences:
            if tid not in prefix_dict:
                prefix_dict[tid] = []
            prefix_dict[tid].append(pos)
        
        for tid, pos in item_occurrences:
            if tid in prefix_dict:
                prefix_positions = prefix_dict[tid]
                for prefix_pos in prefix_positions:
                    if pos > prefix_pos:  
                        new_occurrences.append((tid, pos))
                        break
        
        return new_occurrences

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
    execution_times = []
    for _ in range(5):
        start_time = time.time()
        s = Spade(pos_filepath, neg_filepath, k)
        result = s.min_top_k()
        end_time = time.time()
        execution_times.append(end_time - start_time)
    mean_execution_time = sum(execution_times) / len(execution_times)
    print(f"Mean execution time over 5 runs: {mean_execution_time} seconds")
    return result

if __name__ == '__main__':
    pos_filepath = sys.argv[1]
    neg_filepath = sys.argv[2]
    k = int(sys.argv[3])
    main()