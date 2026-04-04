from pm4py.objects.process_tree.obj import ProcessTree, Operator

def get_base_case(log, empty_cases=0, TAU_THRESHOLD=0.1, LOOP_THRESHOLD=0.2):
  """
  Return the Inductive Miner base case of the log if it is a base case
  :param log: the log to be checked
  :return: the base case if it is a base case, None otherwise
  """
  # If the log is empty return an empty Process Tree
  if log is None or log.empty:
    return ProcessTree()

  # If no empty cases only one activity class and no case is longer than 1 return a Process Tree with a single activity
  if empty_cases == 0 and log["concept:name"].nunique() == 1 and all(log.groupby("case:concept:name").size() == 1):
    return ProcessTree(label=log["concept:name"].iloc[0], children=[])

  if log["concept:name"].nunique() == 1 and all(log.groupby("case:concept:name").size() == 1):
    if empty_cases / (log["case:concept:name"].nunique() + empty_cases) > TAU_THRESHOLD:
      return ProcessTree(operator=Operator.XOR, children=[ProcessTree(label=log["concept:name"].iloc[0]), ProcessTree()])
    return ProcessTree(label=log["concept:name"].iloc[0], children=[])
  
  if log["concept:name"].nunique() == 1:
    if len([entry for entry in log.groupby(["case:concept:name"]).count()["concept:name"].values if entry > 1]) / (log["case:concept:name"].nunique()) > LOOP_THRESHOLD:
      if empty_cases / (log["case:concept:name"].nunique() + empty_cases) > TAU_THRESHOLD:
        return ProcessTree(operator=Operator.LOOP, children=[ProcessTree(), ProcessTree(label=log["concept:name"].iloc[0])])
      return ProcessTree(operator=Operator.LOOP, children=[ProcessTree(label=log["concept:name"].iloc[0]), ProcessTree()])
    elif empty_cases / (log["case:concept:name"].nunique() + empty_cases) > TAU_THRESHOLD:
      return ProcessTree(operator=Operator.XOR, children=[ProcessTree(label=log["concept:name"].iloc[0]), ProcessTree()])
    return ProcessTree(label=log["concept:name"].iloc[0], children=[])

  return None
