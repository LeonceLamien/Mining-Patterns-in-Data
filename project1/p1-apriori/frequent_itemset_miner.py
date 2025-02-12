from itertools import chain, combinations

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

root = Node(0,None,0)
def insert(item,frequency):
    global root
    insert_impl(root, item, 0, None,frequency)

def generate_candidate(frequent_itemsets):
    result = list(all_combinations(frequent_itemsets.keys()))
    return result

def print_trie(node, prefix=[]):
    if node.isKey:
        print(prefix, f"({node.frequency})")

    for num, child in node.next.items():
        print_trie(child, prefix + [num])
        
def is_subset(candidat, itemset):
    candidat_set = set(candidat)
    itemset_set = set(itemset) 
    return candidat_set.issubset(itemset_set)

def all_combinations(iterable,k=0):
    items = list(iterable)
    return chain.from_iterable(combinations(items, r) for r in range(len(items) + 1))

def apriori(filepath, minFrequency):
    """Runs the apriori algorithm on the specified file with the given minimum frequency"""
    frequent_itemsets = {}
    dataset = Dataset(filepath)
    min_count = minFrequency * dataset.trans_num() 
    level = [[] for _ in range(len(dataset.items) + 1)]
    for item in dataset.items:
        counter = sum(1 for transaction in dataset.transactions if item in transaction)
        if counter >= min_count:
            level[1].append(item)
            frequent_itemsets[item] = counter

    k = 2
    while len(level[k - 1]):  
        candidates = generate_candidate(frequent_itemsets)
        frequent_candidates = []

        for item in candidates:
            counter = sum(1 for transaction in dataset.transactions if is_subset(item, transaction))
            if counter >= min_count:
                frequent_candidates.append(item)
                frequent_itemsets[item] = counter

        for item in frequent_candidates:
            if(len(item)==0):continue
            insert(list(item), frequent_itemsets[item]/dataset.trans_num())

        k += 1
    print_trie(root)
    return root
    
def alternative_miner(filepath, minFrequency):
	"""Runs the alternative frequent itemset mining algorithm on the specified file with the given minimum frequency"""
	# TODO: either second implementation of the apriori algorithm or implementation of the depth first search algorithm
	print("Not implemented")

apriori("/Users/leoncelamien/Documents/master/Q2/mining_pattern/LINFO2364---Mining-Patterns-in-Data/project1/p1-apriori/Datasets/toy/toy.dat", 0.125)