#!/usr/bin/env python3

import sys

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
        
        # Find all patterns with dynamic pruning
        all_patterns = []
        
        # Process single-item patterns
        for item in all_unique_items:
            positive_support = len(positive_spade['covers'].get(item, set()))
            negative_support = len(negative_spade['covers'].get(item, set()))
            total_support = positive_support + negative_support
            
            pattern = [item]
            all_patterns.append((total_support, pattern, positive_support, negative_support))
        
        # Sort patterns to get initial min_support
        all_patterns.sort(key=lambda x: (-x[0], x[1]))
        
        # Find longer patterns using DFS
        self.depth_first_search(positive_spade, negative_spade, all_unique_items, all_patterns)
        
        # Final sort of all patterns
        all_patterns.sort(key=lambda x: (-x[0], x[1]))
        
        # Get top-k support values
        support_values = []
        for sup, _, _, _ in all_patterns:
            if sup not in support_values:
                support_values.append(sup)
                if len(support_values) == self.k:
                    break
        
        # If we have fewer than k distinct supports, use the lowest one
        min_support = support_values[-1] if support_values else 0
        
        result = []
        # Select patterns with top-k supports
        for total_support, pattern, pos_support, neg_support in all_patterns:
            if total_support >= min_support:
                result.append((total_support, pattern, pos_support, neg_support))
            else:
                break
        
        # Print the result
        #for total_support, pattern, pos_support, neg_support in result:
        #    pattern_str = '[' + ', '.join(pattern) + ']'
        #    print(f"{pattern_str} {pos_support} {neg_support} {total_support}")
            
        return result

    def depth_first_search(self, positive_spade, negative_spade, items, all_patterns):
        """
        Depth-first search to find all sequential patterns with dynamic pruning
        using early minimum support calculation
        """
        # Calculate initial minimum support threshold based on existing patterns
        min_support = 0
        if len(all_patterns) >= self.k:
            # Sort patterns by support (descending)
            all_patterns.sort(key=lambda x: (-x[0], x[1]))
            
            # Get distinct support values
            support_values = []
            for sup, _, _, _ in all_patterns:
                if sup not in support_values:
                    support_values.append(sup)
                    if len(support_values) == self.k:
                        break
            
            # Use kth support value as minimum threshold
            if len(support_values) == self.k:
                min_support = support_values[-1]
        
        # Stack for DFS
        stack = []
        
        # Initialize the stack with single items
        for item in sorted(items):
            pos_support = len(positive_spade['covers'].get(item, set()))
            neg_support = len(negative_spade['covers'].get(item, set()))
            total_support = pos_support + neg_support
            
            # Only add to stack if support meets minimum threshold
            if total_support >= min_support:
                stack.append(([item], 
                        positive_spade['repr'].get(item, []), 
                        negative_spade['repr'].get(item, [])))
        
        # Track patterns we've added during this DFS
        dfs_patterns_added = 0
        
        # Process patterns in the stack
        while stack:
            pattern, pos_occurrences, neg_occurrences = stack.pop()
            
            # Periodically update min_support (every 100 new patterns)
            if dfs_patterns_added > 0 and dfs_patterns_added % 100 == 0:
                # Recalculate min_support based on latest patterns
                support_values = []
                all_patterns.sort(key=lambda x: (-x[0], x[1]))
                for sup, _, _, _ in all_patterns:
                    if sup not in support_values:
                        support_values.append(sup)
                        if len(support_values) == self.k:
                            break
                
                if len(support_values) == self.k:
                    min_support = support_values[-1]
            
            # Try extending the pattern with each item
            for item in sorted(items):
                # Create a new pattern by appending the item
                new_pattern = pattern + [item]
                
                # Get occurrences of the extended pattern
                new_pos_occurrences = self.get_new_occurrences(pos_occurrences, positive_spade['repr'].get(item, []))
                new_neg_occurrences = self.get_new_occurrences(neg_occurrences, negative_spade['repr'].get(item, []))
                
                # Calculate the support
                pos_support = len(set(tid for tid, _ in new_pos_occurrences))
                neg_support = len(set(tid for tid, _ in new_neg_occurrences))
                total_support = pos_support + neg_support
                
                # Add the pattern to our results if it meets support threshold
                if total_support >= min_support:
                    all_patterns.append((total_support, new_pattern, pos_support, neg_support))
                    dfs_patterns_added += 1
                    
                    # Add to stack for further extension if there are occurrences
                    if new_pos_occurrences or new_neg_occurrences:
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