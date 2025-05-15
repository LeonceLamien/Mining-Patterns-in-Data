import re
import math
import pandas as pd
import numpy as np
import itertools
from collections import defaultdict
import copy
from collections import defaultdict
import sys

# A class for representing the CPT of a variable


class CPT:

    def __init__(self, head, parents):
        self.head = head  # The variable this CPT belongs to (object)
        self.parents = parents  # Parent variables (objects), in order
        self.entries = {}
        # Entries in the CPT. The key of the dictionnary is an
        # assignment to the parents; the associated value is a dictionnary
        # itself, reflecting one row in the CPT.
        # For a variable that has no parents, the key is the empty tuple.

    # String representation of the CPT according to the BIF format

    def __str__(self):
        comma = ", "
        if len(self.parents) == 0:
            return f"probability ( {self.head.name} ) {{" + "\n" \
                f"  table {comma.join ( map(str,self.entries[tuple()].values () ))};" + "\n" \
                f"}}" + "\n"
        else:
            return f"probability ( {self.head.name} | {comma.join ( [p.name for p in self.parents ] )} ) {{" + "\n" + \
                "\n".join ( [  \
                  f"  ({comma.join(names)}) {comma.join(map(str,values.values ()))};" \
                    for names,values in self.entries.items () \
                ] ) + "\n}\n" 


# A class for representing a variable
class Variable:

    def __init__(self, name, values):
        self.name = name  # Name of the variable
        self.values = values
        # The domain of the variable: names of the values
        self.cpt = None  # No CPT initially

    # String representation of the variable according to the BIF format
    def __str__(self):
        comma = ", "
        return f"variable {self.name} {{" + "\n" \
             + f"  type discrete [ {len(self.values)} ] {{ {(comma.join(self.values))} }};" + "\n" \
             + f"}}" + "\n"


class BayesianNetwork:

    # Method for reading a Bayesian Network from a BIF file;
    # fills a dictionary 'variables' with variable names mapped to Variable
    # objects having CPT objects.
    def __init__(self, input_file):

        with open(input_file) as f:
            lines = f.readlines()

        self.variables = {}
        # The dictionary of variables, allowing quick lookup from a variable
        # name.
        for i in range(len(lines)):
            lines[i] = lines[i].lstrip().rstrip().replace('/', '-')

        # Parsing all the variable definitions
        i = 0
        while i < len(lines) and not lines[i].startswith("probability"):
            if lines[i].startswith("variable"):
                variable_name = lines[i].rstrip().split(' ')[1]
                i += 1
                variable_def = lines[i].rstrip().split(' ')
                # only discrete BN are supported
                assert (variable_def[1] == 'discrete')
                variable_values = [x for x in variable_def[6:-1]]
                for j in range(len(variable_values)):
                    variable_values[j] = re.sub(
                        r'\(|\)|,', '', variable_values[j])
                variable = Variable(variable_name, variable_values)
                self.variables[variable_name] = variable
            i += 1

        # Parsing all the CPT definitions
        while i < len(lines):
            if lines[i].startswith('probability'):
                split = lines[i].split(' ')
                target_variable_name = split[2]
                variable = self.variables[target_variable_name]

                parents = [self.variables[x.rstrip().lstrip().replace(',', '')]
                           for x in split[4:-2]]

                assert (variable.name == split[2])

                cpt = CPT(variable, parents)
                i += 1

                if len(parents) > 0:
                    nb_lines = 1
                    for p in parents:
                        nb_lines *= len(p.values)

                    for _ in range(nb_lines):
                        cpt_line = [x for x in re.sub(
                            r'\(|\)|,', '', lines[i][:-1]).split()]
                        parent_values = tuple(
                            [x for x in cpt_line[:len(parents)]])
                        probabilities = [float(p)
                                         for p in cpt_line[len(parents):]]
                        cpt.entries[parent_values] = {
                            v: p for v, p in zip(variable.values, probabilities)}
                        i += 1
                else:
                    cpt_line = [x for x in re.sub(
                        r'\(|\)|,', '', lines[i][:-1]).split()]
                    probabilities = [float(p) for p in cpt_line[1:]]
                    cpt.entries[tuple()] = {v: p for v, p in zip(
                        variable.values, probabilities)}

                variable.cpt = cpt
            i += 1

        # Ensure every variable has a CPT (for variables without CPT in BIF)
        for var in self.variables.values():
            if var.cpt is None:
                var.cpt = CPT(var, [])
                var.cpt.entries[tuple()] = {
                    v: 1.0 / len(var.values) for v in var.values}

    # Method for writing a Bayesian Network to an output file
    # In the BayesianNetwork class's write method:
    def write(self, filename):
        with open(filename, "w") as file:
            # Write variable definitions (no parents here)
            for var in self.variables.values():
                file.write(str(var))
            # Write CPTs (parents are referenced here)
            for var in self.variables.values():
                file.write(str(var.cpt))  # Correctly access CPT's parents

    def infer_single_missing(self, assignment, missing_var):
        """
        assignment_with_missing: dict {var_name: value or None}
        - All observed variables are mapped to their (string) value.
        - Only one variable (named missing_var) is missing (set to None).

        missing_var: the name of the variable to impute.

        Returns:
        dist: dict {value: probability} for P(missing_var=value | evidence)
        """
        var = self.variables[missing_var]
        log_probs = defaultdict(float)

        for v in var.values:
            full_assignment = assignment.copy()
            full_assignment[missing_var] = v

            log_p = 0.0
            for v_name, v_obj in self.variables.items():
                parents = v_obj.cpt.parents
                parent_vals = tuple(full_assignment[p.name] for p in parents)
                val = full_assignment[v_name]

                try:
                    p = v_obj.cpt.entries[parent_vals][val]
                    log_p += math.log(p) if p > 0 else -float('inf')
                except KeyError:
                    log_p = -float('inf')
                    break

            log_probs[v] = log_p

        # Conversion en probabilités normalisées
        max_log = max(log_probs.values()) if log_probs else 0
        probs = {k: math.exp(v - max_log) for k, v in log_probs.items()}
        total = sum(probs.values())

        if total == 0:
            return {v: 1/len(var.values) for v in var.values}

        return {k: v/total for k, v in probs.items()}

    def full_joint_prob(self, assignment):
        """
        Compute the joint probability of a complete assignment.
        assignment: dict of {var_name: value}
        """
        prob = 1.0
        for var_name, var in self.variables.items():
            val = assignment[var_name]
            parents = [p.name for p in var.cpt.parents]
            parent_vals = tuple(assignment[p] for p in parents)
            # Get the conditional probability from the CPT
            p = var.cpt.entries[parent_vals][val]
            prob *= p
        return prob

    def infer_two_missing(self, assignment, missing_vars):
        """
        assignment_with_missing: dict of {var_name: value or None}
        missing_vars: list of two variable names that are missing

        Returns: 
        joint_dist, a dict of (x_val, y_val) -> probability
        """
        var1, var2 = missing_vars
        values1 = self.variables[var1].values
        values2 = self.variables[var2].values

        log_joint = defaultdict(float)

        for v1 in values1:
            for v2 in values2:
                full_assignment = assignment.copy()
                full_assignment[var1] = v1
                full_assignment[var2] = v2

                log_p = 0.0
                for v_name, v_obj in self.variables.items():
                    parents = v_obj.cpt.parents
                    parent_vals = tuple(
                        full_assignment[p.name] for p in parents)
                    val = full_assignment[v_name]

                    try:
                        p = v_obj.cpt.entries[parent_vals][val]
                        log_p += math.log(p) if p > 0 else -float('inf')
                    except KeyError:
                        log_p = -float('inf')
                        break

                log_joint[(v1, v2)] = log_p

        # Normalisation
        max_log = max(log_joint.values()) if log_joint else 0
        probs = {k: math.exp(v - max_log) for k, v in log_joint.items()}
        total = sum(probs.values())

        if total == 0:
            uniform = 1 / (len(values1) * len(values2))
            return {(v1, v2): uniform for v1 in values1 for v2 in values2}

        return {k: v/total for k, v in probs.items()}

    def learn_parameters(self, data, alpha=2, shrinkage_k=10):
        for col in data.columns:
            if col in self.variables:
                var = self.variables[col]
                data[col] = data[col].apply(
                    lambda x: to_bn_value(x) if pd.notnull(x) else x)
                data[col] = data[col].apply(
                    lambda x: x if x in var.values else np.nan)

        for var_name, var in self.variables.items():
            parents = var.cpt.parents
            parent_names = [p.name for p in parents]
            required_columns = [var_name] + parent_names
            data_var = data.dropna(subset=required_columns).copy()

            global_counts = data_var[var_name].value_counts().to_dict()
            global_total = sum(global_counts.values()) + \
                alpha * len(var.values)
            global_probs = {
                val: (global_counts.get(val, 0) + alpha) / global_total
                for val in var.values
            }

            parent_combinations = list(itertools.product(
                *[p.values for p in parents])) if parents else [()]

            for parent_comb in parent_combinations:
                if parents:
                    mask = pd.Series(True, index=data_var.index)
                    for idx, parent_val in enumerate(parent_comb):
                        mask &= (data_var[parent_names[idx]] == parent_val)
                    subset = data_var[mask]
                else:
                    subset = data_var

                n_observed = len(subset)
                if n_observed == 0:
                    var.cpt.entries[parent_comb] = global_probs
                    continue

                value_counts = subset[var_name].value_counts().to_dict()
                total_local = n_observed + alpha * len(var.values)
                local_probs = {
                    val: (value_counts.get(val, 0) + alpha) / total_local
                    for val in var.values
                }

                weight = n_observed / (n_observed + shrinkage_k)
                shrinked_probs = {
                    val: weight * local_probs[val] +
                    (1 - weight) * global_probs[val]
                    for val in var.values
                }
                var.cpt.entries[parent_comb] = shrinked_probs

    def remove_all_edges(self):
        """Set all variables to have no parents (fully disconnected)."""
        for var in self.variables.values():
            var.cpt.parents = []
            var.cpt.entries = {}


def is_missing(val):
    return pd.isnull(val) or val == '' or str(val).lower() in {'nan', 'none'}


def to_bn_value(val):
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    val_clean = str(val).strip().lower()  # Normalisation
    return val_clean


def test_inference_accuracy(bn, test_missing_csv, test_full_csv):
    """
    bn: your BayesianNetwork object, must have infer_single_missing & infer_two_missing methods
    test_missing_csv: path to the CSV with missing values
    test_full_csv:    path to the ground-truth complete CSV

    Outputs row-by-row predictions and accuracy.
    """
    df_missing = pd.read_csv(test_missing_csv)
    df_full = pd.read_csv(test_full_csv)

    # Ensure index alignment (assume both CSVs have same row order)
    headers = list(df_missing.columns)

    n_total = 0
    n_correct = 0
    n_single = 0
    n_two = 0

    for idx, row in df_missing.iterrows():
        assignment = {}
        missing_vars = []

        for var in headers:
            val = row[var]
            if is_missing(val):
                missing_vars.append(var)
                assignment[var] = None
            else:
                assignment[var] = to_bn_value(val)

        if len(missing_vars) == 1:
            missing_var = missing_vars[0]
            prob_dist = bn.infer_single_missing(assignment, missing_var)
            pred_val = max(prob_dist, key=prob_dist.get)
            true_val = to_bn_value(df_full.loc[idx, missing_var])

            # print(f"Row {idx}: Impute {missing_var}. Pred: {pred_val}, True: {true_val}, Prob Dist: {prob_dist}")
            n_total += 1
            n_single += 1
            if pred_val == true_val:
                n_correct += 1

        elif len(missing_vars) == 2:
            var1, var2 = missing_vars
            joint_dist = bn.infer_two_missing(assignment, missing_vars)
            (v1, v2) = max(joint_dist, key=joint_dist.get)
            true1 = to_bn_value(df_full.loc[idx, var1])
            true2 = to_bn_value(df_full.loc[idx, var2])

            # print(f"Row {idx}: Impute {var1},{var2}. Pred: {v1},{v2}, True: {true1},{true2}, Joint Dist: {joint_dist}")
            n_total += 2
            n_two += 1
            if v1 == true1:
                n_correct += 1
            if v2 == true2:
                n_correct += 1

        elif len(missing_vars) == 0:
            # No missing variable, nothing to predict.
            continue

        else:
            # print(f"Row {idx}: Skipping row with >2 missing values: {missing_vars}")
            continue
    return n_correct / n_total if n_total else 0.0


def copy_bn_structure(bn, remove_all_edges=False):
    """Return a new BN with same variables, (optionally) without parent edges."""
    new_bn = BayesianNetwork.__new__(BayesianNetwork)
    new_bn.variables = {}
    # Deep copy Variable objects and empty CPTs
    for name, var in bn.variables.items():
        new_var = Variable(var.name, var.values.copy())
        # Copy CPT object (but will rewrite parents below)
        new_var.cpt = CPT(new_var, [])
        new_bn.variables[name] = new_var

    # Now assign parent pointers
    for name, var in bn.variables.items():
        if not remove_all_edges:
            orig_parents = var.cpt.parents
        else:
            orig_parents = []
        new_parents = [new_bn.variables[p.name] for p in orig_parents]
        new_bn.variables[name].cpt.parents = new_parents

    return new_bn


def compute_mutual_information(data, var1, var2):
    """Compute mutual information between two variables in the dataset."""
    subset = data[[var1, var2]].dropna()
    if len(subset) < 2:
        return 0.0  # Not enough data

    # Compute joint and marginal probabilities
    p_xy = pd.crosstab(subset[var1], subset[var2], normalize=True)
    p_x = subset[var1].value_counts(normalize=True)
    p_y = subset[var2].value_counts(normalize=True)

    mi = 0.0
    for x_val in p_xy.index:
        for y_val in p_xy.columns:
            p_joint = p_xy.loc[x_val, y_val]
            p_x_marginal = p_x.get(x_val, 1e-9)  # Avoid zero
            p_y_marginal = p_y.get(y_val, 1e-9)
            if p_joint > 0:
                mi += p_joint * np.log(p_joint / (p_x_marginal * p_y_marginal))
    return mi


def structure_learning(train_csv, original_bn):
    """Learn a tree-structured BN using Chow-Liu algorithm."""
    data = pd.read_csv(train_csv)
    bn = copy_bn_structure(original_bn, remove_all_edges=True)
    variables = list(bn.variables.keys())

    # Compute mutual information for all pairs
    edges = []
    for i in range(len(variables)):
        var1 = variables[i]
        for j in range(i+1, len(variables)):
            var2 = variables[j]
            mi = compute_mutual_information(data, var1, var2)
            edges.append((-mi, var1, var2))  # Negative for min-heap sorting

    # Kruskal's algorithm for Maximum Spanning Tree
    edges.sort()  # Sorts ascending based on -mi (equiv to descending MI)
    parent = {var: var for var in variables}
    mst_edges = []

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # Path compression
            u = parent[u]
        return u

    for edge in edges:
        mi = -edge[0]
        u, v = edge[1], edge[2]
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            mst_edges.append((u, v))
            parent[root_v] = root_u
        if len(mst_edges) == len(variables) - 1:
            break

    # Build adjacency list for MST
    adj = defaultdict(list)
    for u, v in mst_edges:
        adj[u].append(v)
        adj[v].append(u)

    # Assign directions from root (choose first variable)
    root = variables[0]
    parent_map = {}
    visited = set([root])
    queue = [root]
    while queue:
        node = queue.pop(0)
        for neighbor in adj[node]:
            if neighbor not in visited:
                parent_map[neighbor] = node
                visited.add(neighbor)
                queue.append(neighbor)

    # Update BN structure
    for child, parent_var in parent_map.items():
        # Assign parent to the CPT, not the Variable
        child_var = bn.variables[child]
        parent_obj = bn.variables[parent_var]
        child_var.cpt.parents = [parent_obj]

    # Learn parameters with Dirichlet smoothing
    bn.learn_parameters(data)
    return bn


def main():
    if len(sys.argv) != 5:
        print("Usage: python script.py <training_data> <test_data> <missing_output> <bn_output>")
        sys.exit(1)

    train_data_path, test_data_path, missing_output, bn_output = sys.argv[1:5]
    base_bn = BayesianNetwork(missing_output)
    learned_bn = structure_learning(train_data_path, base_bn)
    learned_bn.write(bn_output)

if __name__ == "__main__":
    main()
    