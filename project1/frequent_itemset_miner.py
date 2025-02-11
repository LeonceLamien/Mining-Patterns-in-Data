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

The items in an itemset must be printed in lexicographical order. However, the itemsets themselves can be printed in
any order.

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
	
def get_patterns_from_file(filename):
	patterns = set()
	has_errors = False
	with open(filename) as f:
		for line in f:
			if line != "":
				g = re.search("\[((?:\d+,? ?)+)\] *(\(\d.\d+\))", line.rstrip())
				if g is None:
					has_errors = True
					print(f"[ERROR] The following line, from file {filename}, has the wrong format:")
					print(f"\t{line}")
				else:
					itemset = tuple(sorted([int(x) for x in g.group(1).split(', ')]))
					patterns.add(itemset)
	return patterns if not has_errors else None
    
def compare_solution_files(expected_file, actual_file):
	"""Compare the output of the patterns in actual with the patterns in expected"""
	expected_patterns = get_patterns_from_file(expected_file)
	actual_patterns = get_patterns_from_file(actual_file)
	if expected_patterns is not None and actual_patterns is not None:
		missed = expected_patterns - actual_patterns
		excess = actual_patterns - expected_patterns
		if len(missed) == 0 and len(excess) == 0:
			print("The files contain the same patterns")
		else:
			if len(missed) != 0:
				print("You missed some itemsets from the expected files:")
				to_show = list(missed)[:10] if len(missed) > 10 else list(missed)
				for pattern in to_show:
					print(f"\t{pattern}")
				print(f"(Showed {len(to_show)} out of {len(missed)})")
			if len(excess) != 0:
				print("You returned unfrequent itemset:")
				to_show = list(excess)[:10] if len(excess) > 10 else list(excess)
				for pattern in to_show:
					print(f"\t{pattern}")
				print(f"(Showed {len(to_show)} out of {len(excess)})")

def apriori(filepath, minFrequency):
	"""Runs the apriori algorithm on the specified file with the given minimum frequency"""
	# TODO: implementation of the apriori algorithm
	dataset = Dataset(filepath)

	print(dataset.items)
	print("Not implemented")


def alternative_miner(filepath, minFrequency):
	"""Runs the alternative frequent itemset mining algorithm on the specified file with the given minimum frequency"""
	# TODO: either second implementation of the apriori algorithm or implementation of the depth first search algorithm
	print("Not implemented")

test = apriori("/Users/leoncelamien/Documents/master/Q2/mining_pattern/project1/p1-apriori/Datasets/mushroom/mushroom.dat", 0.8)
