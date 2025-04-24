import re
import pandas as pd

# A class for representing the CPT of a variable
class CPT:

    def __init__(self, head, parents):
        self.head = head # The variable this CPT belongs to (object)
        self.parents = parents # Parent variables (objects), in order
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
        self.name = name # Name of the variable
        self.values = values 
          # The domain of the variable: names of the values
        self.cpt = None # No CPT initially

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
                assert(variable_def[1] == 'discrete')
                variable_values = [x for x in variable_def[6:-1]]
                for j in range(len(variable_values)):
                    variable_values[j] = re.sub(r'\(|\)|,', '', variable_values[j])
                variable = Variable(variable_name, variable_values)
                self.variables[variable_name] = variable
            i += 1

        
        # Parsing all the CPT definitions
        while i < len(lines):
            if lines[i].startswith('probability'):
                split = lines[i].split(' ')
                target_variable_name = split[2]
                variable = self.variables[target_variable_name]

                parents = [self.variables[x.rstrip().lstrip().replace(',', '')] for x in split[4:-2]]

                assert(variable.name == split[2])

                cpt = CPT(variable, parents)
                i += 1
                
                if len(parents) > 0:
                    nb_lines = 1
                    for p in parents:
                        nb_lines *= len(p.values)

                    for _ in range(nb_lines):
                        cpt_line = [x for x in re.sub(r'\(|\)|,', '', lines[i][:-1]).split()]
                        parent_values = tuple([x for x in cpt_line[:len(parents)]])
                        probabilities = [float(p) for p in cpt_line[len(parents):]]
                        cpt.entries[parent_values] = { v:p for v,p in zip(variable.values,probabilities) }
                        i += 1
                else:
                    cpt_line = [x for x in re.sub(r'\(|\)|,', '', lines[i][:-1]).split()]
                    probabilities = [float(p) for p in cpt_line[1:]]
                    cpt.entries[tuple()] = { v:p for v,p in zip(variable.values,probabilities) }
                    
                variable.cpt = cpt
            i += 1

    # Method for writing a Bayesian Network to an output file
    def write(self,filename):
        with open(filename,"w") as file:
            for var in self.variables.values ():
                file.write(str(var))
            for var in self.variables.values ():
                file.write(str(var.cpt))

    # Example method: returns the probability P(Y=y|X=x),
    # for one variable Y in the BN, y a value in its domain,
    # and x an assignment to its parents in the network, specified
    # in the correct order of parents.
    def P_Yisy_given_parents_x(self,Y,y,x=tuple()):
        return self.variables[Y].cpt.entries[x][y]

    # Example method: returns the probability P(Y=y|X=x),
    # for one variable Y in the BN, y a value in its domain,
    # and pa a dictionnary of assignments to the parents,
    # with for every parent variable its associated assignment.
    def P_Yisy_given_parents(self,Y,y,pa={}):
        x = tuple([ pa[parent.name] for parent in self.variables[Y].cpt.parents ])
        return self.P_Yisy_given_parents_x(Y,y,x)
    
    def infer_single_missing(self, assignment_with_missing, missing_var):
        """
        assignment_with_missing: dict {var_name: value or None}
        - All observed variables are mapped to their (string) value.
        - Only one variable (named missing_var) is missing (set to None).

        missing_var: the name of the variable to impute.

        Returns:
        dist: dict {value: probability} for P(missing_var=value | evidence)
        """
        # Debug: print input assignment for confirmation
        # print("Input assignment:", assignment_with_missing)
        # print("Missing variable:", missing_var)
        
        var = self.variables[missing_var]
        result = {}

        # Will store {v: p} before normalization
        raw_probs = {}

        # For each possible value of the missing variable
        for v in var.values:
            # Copy evidence assignment and fill in the missing var
            full_assignment = assignment_with_missing.copy()
            full_assignment[missing_var] = v

            # Debug: print full assignment for this trial
            # print(f"Trying {missing_var} = {v} -> full assignment:", full_assignment)

            # Calculate the joint probability of the assignment
            jp = 1.0
            for vn, vobj in self.variables.items():
                # Build parent assignment for this variable
                parents = vobj.cpt.parents
                pa_tuple = tuple(full_assignment[parent.name] for parent in parents)
                val = full_assignment[vn]

                # Sanity check
                if val not in vobj.values:
                    print(f"Warning: {val} not a valid value for {vn}")
                
                prob = vobj.cpt.entries[pa_tuple][val]
                jp *= prob

                # Debug: Print each factor
                # print(f"P({vn}={val} | parents={pa_tuple}) = {prob}")

            # print(f"Joint probability for {missing_var}={v}: {jp}")
            raw_probs[v] = jp

        # Normalize
        total = sum(raw_probs.values())
        if total == 0:
            print("WARNING: Sum of probabilities is zero! (Check assignments & data)")
            return {v: 0.0 for v in var.values}
        result = {v: p / total for v, p in raw_probs.items()}

        # Debug: print normalized distribution
        # print("Resulting distribution for", missing_var, ":", result)
        return result
    
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

    
    def infer_two_missing(self, assignment_with_missing, missing_vars):
        """
        assignment_with_missing: dict of {var_name: value or None}
        missing_vars: list of two variable names that are missing

        Returns: 
        joint_dist, a dict of (x_val, y_val) -> probability
        """
        var1, var2 = missing_vars
        values1 = self.variables[var1].values 
        values2 = self.variables[var2].values

        joint_probs = {}

        for v1 in values1:
            for v2 in values2:
                # Fill in missing with tested values
                assignment = assignment_with_missing.copy()
                assignment[var1] = v1
                assignment[var2] = v2

                # Compute joint probability of full assignment
                p = self.full_joint_prob(assignment)  # or whatever your method is
                joint_probs[(v1, v2)] = p
                
                # Debug: print each trial
                # print(f"Trying {var1}={v1}, {var2}={v2} -> prob={p}")

        # Normalize
        total = sum(joint_probs.values())
        if total > 0:
            for k in joint_probs:
                joint_probs[k] /= total

        # Debug: print normalized joint distribution
        # print(f"Resulting joint for {var1},{var2} :", joint_probs)
        return joint_probs

    
def is_missing(val):
    return pd.isnull(val) or val == '' or str(val).lower() in {'nan', 'none'}

def to_bn_value(val):
    # Adjust as needed to match BN's expected format!
    # Here, ensuring string type and handling float->int if needed
    try:
        return str(int(float(val)))
    except (ValueError, TypeError):
        return str(val) if val is not None else None

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
            if pred_val == true_val: n_correct += 1

        elif len(missing_vars) == 2:
            var1, var2 = missing_vars
            joint_dist = bn.infer_two_missing(assignment, missing_vars)
            (v1, v2) = max(joint_dist, key=joint_dist.get)
            true1 = to_bn_value(df_full.loc[idx, var1])
            true2 = to_bn_value(df_full.loc[idx, var2])

            # print(f"Row {idx}: Impute {var1},{var2}. Pred: {v1},{v2}, True: {true1},{true2}, Joint Dist: {joint_dist}")
            n_total += 2
            n_two += 1
            if v1 == true1: n_correct += 1
            if v2 == true2: n_correct += 1

        elif len(missing_vars) == 0:
            # No missing variable, nothing to predict.
            continue

        else:
            # print(f"Row {idx}: Skipping row with >2 missing values: {missing_vars}")
            continue

    print(f"Total missing values predicted: {n_total}")
    print(f"Correct predictions: {n_correct}")
    print(f"Accuracy: {n_correct / n_total if n_total else 0:.4f}")
    print(f"Rows with 1 missing: {n_single}, with 2 missing: {n_two}, skipped: {len(df_missing) - n_single - n_two}")

    return n_correct / n_total if n_total else 0.0

#bn = BayesianNetwork("datasets/alarm/alarm_complete.bif")
#test_inference_accuracy(bn, "datasets/alarm/test_missing.csv", "datasets/alarm/test.csv")

bn = BayesianNetwork("datasets/andes/andes_complete.bif")
test_inference_accuracy(bn, "datasets/andes/test_missing.csv", "datasets/andes/test.csv")