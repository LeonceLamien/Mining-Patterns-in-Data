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
				self.transactions.append(line.replace(" ", ""))
				transaction = list(map(int, line.split(" ")))
				for item in transaction:
					self.items.add(str(item))
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

def char_to_index(c):
    return ord(c) - ord('1')

def insert_impl(node, word, d, parent,frequency):
    if node is None:
        node = Node(frequency,word[d - 1], parent)

    if d == len(word):
        node.isKey = True
        return node

    c = word[d]
    if c not in node.next:
        node.next[c] = Node(frequency,c, node)

    node.next[c] = insert_impl(node.next[c], word, d + 1, node,frequency)
    return node


root = Node(0,None,0)
def insert(item,frequency):
    global root
    insert_impl(root, item, 0, None,frequency)


def get_support(root, item):
    node = root
    for c in item:
        if c not in node.next:
            return 0 
        node = node.next[c]
    return node.count


def generate_candidate(prev_frequent_itemsets, k):
    candidates = set()
    prev_frequent_itemsets = sorted(prev_frequent_itemsets)  

    for i in range(len(prev_frequent_itemsets)):
        for j in range(i + 1, len(prev_frequent_itemsets)):
            itemset1 = prev_frequent_itemsets[i]
            itemset2 = prev_frequent_itemsets[j]

            prefix1 = itemset1[:-1]
            prefix2 = itemset2[:-1]

            if prefix1 == prefix2:
                new_candidate = ''.join(sorted(itemset1 + itemset2[-1])) 
                if len(new_candidate) == k:
                    candidates.add(new_candidate)

    return list(candidates)

def prune_candidate(root, candidates, min_frequency):
    frequent_itemsets = []
    for item in candidates:
        support_candidate = get_support(root, item)
        if support_candidate >= min_frequency:
            frequent_itemsets.append(item)
    return frequent_itemsets

def print_trie(node, prefix=""):
    if node.isKey:
        print(f"[{', '.join(prefix)}] ({node.frequency})")

    for key, child in node.next.items():
        print_trie(child, prefix + key)
        
def apriori(filepath, minFrequency):
    """Runs the apriori algorithm on the specified file with the given minimum frequency"""
    dataset = Dataset(filepath)
    min_count = minFrequency * dataset.trans_num() 
    level = [[] for _ in range(len(dataset.items) + 1)]

    item_counts = {}
    for transaction in dataset.transactions:
        for item in set(transaction):  
            item_counts[item] = item_counts.get(item, 0) + 1

    for item, count in item_counts.items():
        if count >= min_count:
            level[1].append(item)
            insert(item,count/dataset.trans_num())

    k = 2
    while len(level[k - 1]):  
        candidates = generate_candidate(level[k - 1], k)
        frequent_candidates = []

        for item in candidates:
            counter = sum(1 for transaction in dataset.transactions if set(item).issubset(transaction))
            if counter >= min_count:
                frequent_candidates.append(item)

        for item in frequent_candidates:
            insert(item,counter/dataset.trans_num())
            level[k].append(item)

        k += 1
    print_trie(root)
    return root
    
def alternative_miner(filepath, minFrequency):
	"""Runs the alternative frequent itemset mining algorithm on the specified file with the given minimum frequency"""
	# TODO: either second implementation of the apriori algorithm or implementation of the depth first search algorithm
	print("Not implemented")


test = apriori("/Users/leoncelamien/Documents/master/Q2/mining_pattern/project1/p1-apriori/Datasets/toy/toy.dat", 0.125)
