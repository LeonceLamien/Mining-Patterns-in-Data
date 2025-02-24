from collections import defaultdict

"""
Skeleton file for the project 1 of the LINGI2364 course.
Use this as your submission file. Every piece of code that is used in your program should be put inside this file.

This file given to you as a skeleton for your implementation of the Apriori and Depth
First Search algorithms. You are not obligated to use them and are free to write any class or method as long as the
following requirements are respected:

Your apriori and alternativeMiner methods must take as parameters a string corresponding to the path to a valid
dataset file and a double corresponding to the minimum frequency.
You must write on the standard output (use the print() method) all the itemsets that are frequent in the dataset file
according to the minimum frequency given. Each itemset has to be printed on one line following the format:
[<item 1>, <item 2>, ... <item k>] (<frequency>).

Do not change the signature of the apriori and alternative_miner methods as they will be called by the test script.

__authors__ = "<write here your group, first name(s) and last name(s)>"
"""

class Dataset:
	"""Utility class to manage a dataset stored in a external file."""
	def __init__(self, filepath):
		"""reads the dataset file and initializes files"""
		self.transactions = list()
		self.items = set()

		try:
			lines = [line.strip() for line in open(filepath, "r")]
			lines = [line for line in lines if line]  # Skipping blank lines
			for line in lines:
				transaction = list(map(int, line.split(" ")))
				self.transactions.append(transaction)
				for item in transaction:
					self.items.add(item)
		except IOError as e:
			print("Unable to read dataset file!\n" + e)

	def trans_num(self):
		"""Returns the number of transactions in the dataset"""
		return len(self.transactions)

	def items_num(self):
		"""Returns the number of different items in the dataset"""
		return len(self.items)

	def get_transaction(self, i):
		"""Returns the transaction at index i as an int array"""
		return self.transactions[i]
	

class Node:
    def __init__(self,frequency, item=None, parent=None):
        self.item = item
        self.next = {}
        self.count = 1
        self.parent = parent
        self.frequency=frequency
        self.isKey = False

    def __str__(self):
        return str(self.item) + " (" + str(self.count) + ")"

def insert_impl(node, numbers, d, parent,fre):
    if d == len(numbers):  
        node.isKey = True
        return node

    num = numbers[d]
    if num not in node.next:
        node.next[num] = Node(fre,num, node)

    node.next[num] = insert_impl(node.next[num], numbers, d + 1, node,fre)
    return node


def insert(root,item,frequency):
    # global root
    insert_impl(root, item, 0, None,frequency)


def prune_candidate(prev_frequent_itemsets):
    """ Generates candidates of size k from frequent itemsets of size k-1 """
    sorted_items = sorted(prev_frequent_itemsets)
    result = []
    for i in range(len(sorted_items)):
        for j in range(i + 1, len(sorted_items)):
            item1 = sorted_items[i]
            item2 = sorted_items[j]
            if has_common_prefix(item1, item2):
                new_candidate = merge_items(item1, item2)
                result.append(new_candidate)
    return result


def has_common_prefix(itemset1, itemset2):
    """ Verify if two itemsets have a common prefix """
    itemset1 = sorted(itemset1)
    itemset2 = sorted(itemset2)
    if len(itemset1) != len(itemset2):  
        return False
    return itemset1[:-1] == itemset2[:-1] 


def merge_items(itemset1, itemset2):
    """ Merge two itemsets to create a new candidate """
    return tuple(sorted(set(itemset1) | set(itemset2)))


def print_trie(node, prefix=[]):
    if node.isKey:
        print(prefix, f"({node.frequency})")

    for num, child in node.next.items():
        print_trie(child, prefix + [num])
        
def is_subset(candidat, itemset):
    candidat_set = set(candidat)
    itemset_set = set(itemset) 
    return candidat_set.issubset(itemset_set)

def prune_candidate(prev_frequent_itemsets):
    sorted_items = sorted(prev_frequent_itemsets)
    result = []
    frequent_set = set(prev_frequent_itemsets)  

    for i in range(len(sorted_items)):
        for j in range(i + 1, len(sorted_items)):
            if has_common_prefix(sorted_items[i], sorted_items[j]):
                new_candidate = merge_items(sorted_items[i], sorted_items[j])
                if new_candidate not in frequent_set: 
                    result.append(new_candidate)
    return result


def count_candidates(candidates, transactions):
    candidate_counts = defaultdict(int)
    transactions_set = [set(transaction) for transaction in transactions]  

    for candidate in candidates:
        candidate_set = set(candidate)
        for transaction_set in transactions_set:
            if candidate_set.issubset(transaction_set):
                candidate_counts[candidate] += 1

    return candidate_counts


def apriori(filepath, minFrequency):
    """Runs the apriori algorithm on the specified file with the given minimum frequency"""
    dataset = Dataset(filepath)
    min_count = minFrequency * dataset.trans_num()
    root = Node(0,None,0)

    item_counts = defaultdict(int)
    for transaction in dataset.transactions:
        for item in transaction:
            item_counts[(item,)] += 1  

    frequent_itemsets = {item: count for item, count in item_counts.items() if count >= min_count}
    
    for item, count in frequent_itemsets.items():
        insert(root,item, count / dataset.trans_num())

    k = 2
    current_level = list(frequent_itemsets.keys())

    while current_level: 
        candidates = prune_candidate(current_level)
        candidate_counts = count_candidates(candidates, dataset.transactions)

        frequent_candidates = {item: count for item, count in candidate_counts.items() if count >= min_count}
        
        if not frequent_candidates: 
            break

        for item, count in frequent_candidates.items():
            insert(root,list(item), count / dataset.trans_num())

        current_level = list(frequent_candidates.keys()) 

        k += 1

    print_trie(root)
    return root
def clean_projection(full_vertical_db, min_support):
    return {k: v for k, v in full_vertical_db.items() if len(v) >= min_support}

def dfs(root,itemset, full_vertical_db, min_support,cache, trans_num):
    for key, tids in full_vertical_db.items():
        new_itemset = tuple(itemset) + (key,)
        new_itemset_frozen =frozenset(new_itemset)
        support = len(tids)
        if support >= min_support and new_itemset_frozen not in cache:
            cache[new_itemset_frozen] = new_itemset
            insert(root,new_itemset, support / trans_num)
            new_vdb = projection(full_vertical_db, key, min_support)
            if any(len(v) >= min_support for v in new_vdb.values()):
                dfs(root,new_itemset, new_vdb, min_support, cache, trans_num)

def vertical_representation(dataset):
    vertical_db = defaultdict(set)
    for tid, transaction in enumerate(dataset.transactions):
        for item in transaction:
            vertical_db[item].add(tid)
    return vertical_db

def projection(full_vertical_db, item, min_support):
    item_tids = full_vertical_db[item]
    return {
        k: frozenset(v & item_tids)
        for k, v in full_vertical_db.items()
        if k != item and len(v & item_tids) >= min_support
    }



def alternative_miner(filepath, minFrequency):
    """Runs the alternative frequent itemset mining algorithm on the specified file with the given minimum frequency"""
    dataset = Dataset(filepath)
    trans_num = dataset.trans_num()
    root = Node(0,None,0)

    min_support = minFrequency * trans_num
    full_vertical_db = vertical_representation(dataset)
    cache = {}
    sorted_items = sorted(full_vertical_db.items(), key=lambda x: -len(x[1]))
    for key, tids in sorted_items:
        new_itemset = [key]
        if len(tids) >= min_support:
            insert(root,new_itemset, len(tids) / trans_num)
            cache[frozenset(new_itemset)] = new_itemset
            new_vdb = projection(full_vertical_db, key, min_support)
            dfs(root,new_itemset, new_vdb, min_support, cache, trans_num)

    print_trie(root)
    return root
