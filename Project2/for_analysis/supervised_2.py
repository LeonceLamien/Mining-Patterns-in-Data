#!/usr/bin/env python3

import sys
import time

class Spade:

    def __init__(self, pos_filepath, neg_filepath, k):
        self.pos_transactions = get_transactions(pos_filepath)
        self.neg_transactions = get_transactions(neg_filepath)
        self.k = k
    
    def calculate_wracc(self, pos_support, neg_support):
        """
        Calculate the Weighted Relative Accuracy (Wracc) score
        """
        P = len(self.pos_transactions)
        N = len(self.neg_transactions)
        
        if P == 0 or N == 0 or (P + N) == 0:
            return 0
            
        # Calculate Wracc
        factor1 = (P / (P + N)) * (N / (P + N))
        factor2 = (pos_support / P) - (neg_support / N)
        
        return round(factor1 * factor2, 5)
    
    def calculate_upper_bound_wracc(self, pos_support, neg_support):
        """
        Calculate the upper bound for Wracc score of any extension of this pattern.
        Any extension of the current pattern cannot have Wracc higher than this value.
        """
        P = len(self.pos_transactions)
        N = len(self.neg_transactions)
        
        if P == 0 or N == 0 or (P + N) == 0:
            return 0
            
        factor1 = (P / (P + N)) * (N / (P + N))
        
        # Best possible scenario is that the pattern extension will be in all remaining
        # positive transactions and none of the negative ones
        max_pos_diff = pos_support / P
        min_neg_diff = 0 / N  # Best case: no negative coverage
        
        # Calculate best possible Wracc
        max_factor2 = max_pos_diff - min_neg_diff
        
        return factor1 * max_factor2
    
    def min_top_k_wracc(self):
        """
        Mine the top-k sequential patterns using the SPADE algorithm
        and Weighted Relative Accuracy (Wracc) as the scoring function
        """
        
        # Convert transactions to SPADE vertical representation
        positive_spade = spade_repr_from_transaction(self.pos_transactions)
        negative_spade = spade_repr_from_transaction(self.neg_transactions)
    
        # Get all unique items
        all_unique_items = sorted(list(set(positive_spade['repr'].keys()).union(negative_spade['repr'].keys())))
        
        # Pattern scores will store (pattern, pos_support, neg_support, wracc) tuples
        pattern_scores = []
        
        # Start with single-item patterns
        for item in all_unique_items:
            pattern = [item]
            pos_tids = set(tid for tid, _ in positive_spade['repr'].get(item, []))
            neg_tids = set(tid for tid, _ in negative_spade['repr'].get(item, []))
            
            # Skip if the pattern has no support
            if not pos_tids and not neg_tids:
                continue
                
            # Start DFS from this item
            self.dfs_wracc(pattern, pos_tids, neg_tids, positive_spade, 
                          negative_spade, all_unique_items, pattern_scores)
        
        # Sort by Wracc (descending)
        pattern_scores.sort(key=lambda x: (-x[3], x[0]))
        
        # Get top-k distinct Wracc values
        distinct_wracc = []
        min_wracc = float('-inf')
        
        for _, _, _, wracc in pattern_scores:
            if wracc not in distinct_wracc:
                distinct_wracc.append(wracc)
                if len(distinct_wracc) == self.k:
                    min_wracc = wracc
                    break
        
        # Select all patterns with Wracc â‰¥ min_wracc
        result = []
        for pattern, pos_support, neg_support, wracc in pattern_scores:
            if wracc >= min_wracc:
                result.append((pattern, pos_support, neg_support, wracc))
        
        # Print patterns in the required format
        for pattern, pos_support, neg_support, wracc in result:
            pattern_str = '[' + ', '.join(pattern) + ']'
            print(f"{pattern_str} {pos_support} {neg_support} {wracc}")
        
        return result
    
    def dfs_wracc(self, pattern, pos_tids, neg_tids, positive_spade, negative_spade, all_unique_items, pattern_scores):
        """
        Depth-first search to find sequential patterns with high Wracc scores
        """
        # Calculate current Wracc
        pos_support = len(pos_tids)
        neg_support = len(neg_tids)
        wracc = self.calculate_wracc(pos_support, neg_support)
        
        # Add the current pattern to our list
        pattern_scores.append((pattern.copy(), pos_support, neg_support, wracc))
        
        # Calculate the minimum Wracc threshold from the current top-k patterns
        min_wracc = float('-inf')
        if len(pattern_scores) >= self.k:
            # Sort patterns by Wracc score (descending)
            pattern_scores.sort(key=lambda x: -x[3])
            
            # Get k-th distinct Wracc value
            distinct_wracc = []
            for _, _, _, w in pattern_scores:
                if w not in distinct_wracc:
                    distinct_wracc.append(w)
                    if len(distinct_wracc) == self.k:
                        min_wracc = w
                        break
            
        
        # Try extending the pattern with each item (including items already in the pattern)
        for item in all_unique_items:
            new_pattern = pattern + [item]
            
            # Get transaction IDs for extended pattern
            new_pos_tids = self.get_tids_for_extension(pattern, item, pos_tids, positive_spade)
            new_neg_tids = self.get_tids_for_extension(pattern, item, neg_tids, negative_spade)
            
            new_pos_support = len(new_pos_tids)
            new_neg_support = len(new_neg_tids)
            
            if new_pos_support == 0 and new_neg_support == 0:
                continue
            
            # Calculate upper bound
            upper_bound = self.calculate_upper_bound_wracc(new_pos_support, new_neg_support)
            
            # Pruning
            if upper_bound <= min_wracc and min_wracc != float('-inf'):
                continue
            
            # Continue DFS
            self.dfs_wracc(new_pattern, new_pos_tids, new_neg_tids, positive_spade, 
                         negative_spade, all_unique_items, pattern_scores)
    
    def get_tids_for_extension(self, pattern, new_item, current_tids, spade_repr):
        """
        Get transaction IDs that contain the extended pattern (pattern + new_item)
        """
        
        if not pattern:
            # For single item pattern
            result = set(tid for tid, _ in spade_repr['repr'].get(new_item, []))
            return result
        
        if not current_tids:
            return set()
        
        # For pattern extension, find transactions where new_item follows the pattern
        result_tids = set()
        
        for tid in current_tids:
            # Get all positions of the pattern in this transaction
            pattern_positions = self.get_all_pattern_positions(pattern, tid, spade_repr)
            
            # Get all positions of the new item in this transaction
            new_item_positions = [pos for t, pos in spade_repr['repr'].get(new_item, []) if t == tid]
            
            # For each pattern position, check if the new item follows it
            for pattern_end_pos in pattern_positions:
                valid_positions = [pos for pos in new_item_positions if pos > pattern_end_pos]
                if valid_positions:
                    result_tids.add(tid)
                    break
        return result_tids
    
    def get_all_pattern_positions(self, pattern, tid, spade_repr):
        """
        Find all valid ending positions for the pattern in the given transaction
        """
        # For a single item pattern, find all its positions
        if len(pattern) == 1:
            positions = [pos for t, pos in spade_repr['repr'].get(pattern[0], []) if t == tid]
            return positions
        
        # For multi-item patterns, recursively find valid positions
        prev_positions = self.get_all_pattern_positions(pattern[:-1], tid, spade_repr)
        if not prev_positions:
            return []
        
        # Get positions of the last item in the pattern
        last_item_positions = [pos for t, pos in spade_repr['repr'].get(pattern[-1], []) if t == tid]
        
        # Find all valid combinations where last_item follows a prev_position
        valid_end_positions = []
        for prev_pos in prev_positions:
            valid_positions = [pos for pos in last_item_positions if pos > prev_pos]
            valid_end_positions.extend(valid_positions)
        return valid_end_positions
    
    def get_pattern_support(self, pattern, spade_repr):
        """
        Calculate the support (number of transactions) of a pattern
        """
        if not pattern:
            return 0
        
        # Start with all transactions that contain the first item
        tids = set(tid for tid, _ in spade_repr.get(pattern[0], []))
        
        # For single item pattern
        if len(pattern) == 1:
            return len(tids)
        
        # Construct the pattern incrementally
        for i in range(1, len(pattern)):
            # Find transactions where the current item follows the previous items
            new_tids = set()
            for tid in tids:
                # Check if this transaction contains the current item
                item_positions = [pos for t, pos in spade_repr.get(pattern[i], []) if t == tid]
                
                # Get the last position of the prefix pattern
                prefix_last_pos = self.get_last_position_in_transaction(pattern[:i], tid, spade_repr)
                
                # If the item occurs after the prefix, add this transaction
                if any(pos > prefix_last_pos for pos in item_positions):
                    new_tids.add(tid)
            
            tids = new_tids
            if not tids:
                break
                
        return len(tids)
    
    def get_last_position_in_transaction(self, pattern, tid, spade_repr):
        """
        Get the last position of the pattern in a transaction
        """
        if not pattern:
            return -1
            
        if len(pattern) == 1:
            positions = [pos for t, pos in spade_repr.get(pattern[0], []) if t == tid]
            return max(positions) if positions else -1
            
        # Get positions of the prefix
        prefix_positions = self.get_all_pattern_positions(pattern[:-1], tid, spade_repr)
        if not prefix_positions:
            return -1
            
        # Get positions of the last item
        last_item_positions = [pos for t, pos in spade_repr.get(pattern[-1], []) if t == tid]
        
        # Find the last valid position
        max_pos = -1
        for prefix_pos in prefix_positions:
            valid_positions = [pos for pos in last_item_positions if pos > prefix_pos]
            if valid_positions:
                max_pos = max(max_pos, max(valid_positions))
                
        return max_pos


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
    execute_time = []
    for _ in range (5):
        start_time = time.time()
        s = Spade(pos_filepath, neg_filepath, k)
        result = s.min_top_k_wracc()
        end_time = time.time()
        execute_time.append(end_time - start_time)
    print("Mean execution time: ", sum(execute_time) / len(execute_time))
    return result

if __name__ == '__main__':
    pos_filepath = sys.argv[1]
    neg_filepath = sys.argv[2]
    k = int(sys.argv[3])
    main()