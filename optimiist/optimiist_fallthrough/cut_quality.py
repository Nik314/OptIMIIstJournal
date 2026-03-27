from typing import Tuple
import pm4py
from pm4py.objects.process_tree.obj import Operator
import numpy as np

CUSTOM_WEIGHTS = {
  "sequence": 1.0,
  "xor": 1.0,
  "parallel": 1.0,
  "loop": 1.0,
  "tau_loop": 1.0
}

def evalutate_cut(cut, log, log_a, log_b, dfg) -> Tuple[float, float, float]:
  if cut[0] == Operator.SEQUENCE:
    return get_seq_conformance(dfg, cut[1], cut[2], log_a, log_b)
  elif cut[0] == Operator.XOR:
    return get_xor_conformance(dfg, cut[1], cut[2])
  elif cut[0] == Operator.PARALLEL:
    return get_and_conformance(log, log_a, log_b)
  elif cut[0] == Operator.LOOP and len(log_a) > 0 and len(log_b) > 0:
    return get_loop_conformance(log, dfg, cut[1], cut[2], log_a, log_b)
  elif cut[0] == Operator.LOOP and len(log_a) > 0:
    return get_tau_loop_confromance(log, dfg, log_a)
  elif cut[0] == Operator.LOOP and len(log_b) > 0:
    return get_tau_loop_confromance(log, dfg, log_b)
  else:
    raise Exception("Invalid cut: " + str(cut) + " " + str(log_a) + " " + str(log_b) + " " + str(log))

def calculate_mae(truth_array, input_array):
    if len(truth_array) != len(input_array):
        raise ValueError("Arrays must be of the same length")
        
    # Convert lists to numpy arrays if they are not already
    truth_array = np.array(truth_array)
    input_array = np.array(input_array)
        
    # Calculate the Mean Absolute Error
    mae = np.mean(np.abs(truth_array - input_array))
    return mae

def f1_score(precision, fitness):
    return 2 * (precision * fitness) / (precision + fitness)

def get_seq_conformance(dfg, partition_1, partition_2, log_1, log_2) -> Tuple[float, float, float]:
  
  # Fitness
  violating_edges = 0
  crossing_edges = 0
  
  for edge in dfg:
    if edge[0] in partition_1 and edge[1] in partition_2:
      crossing_edges += dfg[edge]
    elif edge[0] in partition_2 and edge[1] in partition_1:
      violating_edges += dfg[edge]
      crossing_edges += dfg[edge]

  if crossing_edges == 0:
    fitness = 1
  else:
    fitness = 1 - (violating_edges / crossing_edges)

  # Precision
  end_1 = pm4py.get_end_activities(log_1)
  start_2 = pm4py.get_start_activities(log_2)

  total_end_activities = 0
  for a in end_1:
    total_end_activities += end_1[a]

  base_probabilities = {}
  actual_probabilities = {}

  for end_activity_1 in end_1:
    total_transitions = 0
    for start_activity_2 in start_2:
      total_transitions += dfg[(end_activity_1, start_activity_2)]
      actual_probabilities[end_activity_1, start_activity_2] = dfg[(end_activity_1, start_activity_2)]
      base_probabilities[end_activity_1, start_activity_2] = start_2[start_activity_2] / total_end_activities
    
    if total_transitions != 0:
      for start_activity_2 in start_2:
        actual_probabilities[end_activity_1, start_activity_2] /= total_transitions
    else:
      for start_activity_2 in start_2:
        actual_probabilities[end_activity_1, start_activity_2] = 0

  precision = 1 - calculate_mae(
    [base_probabilities[a,b] for a in end_1 for b in start_2], 
    [actual_probabilities[a,b] for a in end_1 for b in start_2]
  )
  equality_term = len(partition_1) *len(partition_2) / ((0.5*len(partition_1)+len(partition_2))**2)
  return fitness, precision, f1_score(precision, fitness) * CUSTOM_WEIGHTS["sequence"] * equality_term

def get_xor_conformance(dfg, partition_1, partition_2) -> Tuple[float, float, float]:
  # Fitness
  crossing_edges = 0
  possible_edges = 0

  for activity_1 in partition_1:
    for activity_2 in partition_2:
      possible_edges += 2

  for edge in dfg:
    if edge[0] in partition_1 and edge[1] in partition_2 and dfg[edge] > 0:
      crossing_edges += 1
    elif edge[1] in partition_1 and edge[0] in partition_2 and dfg[edge] > 0:
      crossing_edges += 1

  try:
    fitness = 1 - (crossing_edges / possible_edges)
  except:
    return 0,0,0
  # Precision
  precision = 1

  equality_term = len(partition_1) *len(partition_2) / ((0.5*len(partition_1)+len(partition_2))**2)
  return fitness, precision, f1_score(precision, fitness) * CUSTOM_WEIGHTS["xor"] * equality_term

def get_and_conformance(log, log_a, log_b) -> Tuple[float, float, float]:
  # Fitness
  fitness = 1

  # Precision
  variants_base = pm4py.statistics.variants.log.get.get_variants(log)

  variants_a = pm4py.statistics.variants.log.get.get_variants(log_a)
  variants_b = pm4py.statistics.variants.log.get.get_variants(log_b)

  average_variant_length_a = sum([len(variant) for variant in variants_a]) / len(variants_a) if len(variants_a) > 0 else 0
  average_variant_length_b = sum([len(variant) for variant in variants_b]) / len(variants_b) if len(variants_b) > 0 else 0

  expected_variants = 2**(average_variant_length_a + average_variant_length_b)

  precision = len(variants_base) / expected_variants

  partition_1 = log_a["concept:name"].unique()
  partition_2 = log_b["concept:name"].unique()
  equality_term = len(partition_1) *len(partition_2) / ((0.5*len(partition_1)+len(partition_2))**2)
  return fitness, precision, f1_score(precision, fitness) * CUSTOM_WEIGHTS["parallel"]*equality_term

def get_loop_conformance(log, dfg, partition_1, partition_2, loop_a, loop_b) -> Tuple[float, float, float]:
  # Fitness
  start = pm4py.get_start_activities(log)
  end = pm4py.get_end_activities(log)

  start_activities_in_2 = 0
  end_activities_in_2 = 0

  for activity in start:
    if activity in partition_2:
      start_activities_in_2 += start[activity]

  for activity in end:
    if activity in partition_2:
      end_activities_in_2 += end[activity]

  start_activities_total = sum([start[activity] for activity in start])
  end_activities_total = sum([end[activity] for activity in end])

  fitness = 1 - (start_activities_in_2 + end_activities_in_2) / (start_activities_total + end_activities_total)

  # Precision
  end_1 = pm4py.get_end_activities(loop_a)

  start_2 = pm4py.get_start_activities(loop_b)
  total_start_2_activities = sum([start_2[a] for a in start_2])

  base_enter_probabilities = {}
  actual_enter_probabilities = {}
  
  for end_activity_1 in end_1:
    total_transitions = 0
    for start_activity_2 in start_2:
      total_transitions += dfg[(end_activity_1, start_activity_2)]
      actual_enter_probabilities[(end_activity_1, start_activity_2)] = dfg[(end_activity_1, start_activity_2)]
      base_enter_probabilities[(end_activity_1, start_activity_2)] = start_2[start_activity_2] / total_start_2_activities

    if total_transitions != 0:
      for start_activity_2 in start_2:
        actual_enter_probabilities[(end_activity_1,start_activity_2)] /= total_transitions
    else:
      for start_activity_2 in start_2:
        actual_enter_probabilities[(end_activity_1,start_activity_2)] = 1

  start_1 = pm4py.get_start_activities(loop_a)
  end_2 = pm4py.get_end_activities(loop_b)
  start_full = pm4py.get_start_activities(log)
  total_end_2_activities = sum([end_2[a] for a in end_2])

  base_exit_probabilities = {}
  actual_exit_probabilities = {}

  for end_activity_2 in end_2:
    total_transitions = 0
    for start_activity_1 in start_1:
      total_transitions += dfg[(end_activity_2, start_activity_1)]

      actual_exit_probabilities[(end_activity_2, start_activity_1)] = dfg[(end_activity_2, start_activity_1)]
      base_exit_probabilities[(end_activity_2, start_activity_1)] = (start_1[start_activity_1] - (start_full[start_activity_1] if start_activity_1 in start_full else 0)) / total_end_2_activities

    if total_transitions != 0:
      for start_activity_1 in start_1:
        actual_exit_probabilities[(end_activity_2,start_activity_1)] /= total_transitions
    else:
      for start_activity_1 in start_1:
        actual_exit_probabilities[(end_activity_2,start_activity_1)] = 1

  enter_mae = calculate_mae(
    [base_enter_probabilities[(a,b)] for a in end_1 for b in start_2],
    [actual_enter_probabilities[(a,b)] for a in end_1 for b in start_2]
  )

  exit_mae = calculate_mae(
    [base_exit_probabilities[(a,b)] for a in end_2 for b in start_1], 
    [actual_exit_probabilities[(a,b)] for a in end_2 for b in start_1]
  )

  precision = (1 - (enter_mae + exit_mae) / 2)

  equality_term = len(partition_1) *len(partition_2) / ((0.5*len(partition_1)+len(partition_2))**2)
  return fitness, precision, f1_score(precision, fitness) * CUSTOM_WEIGHTS["loop"]*equality_term

def get_tau_loop_confromance(log, dfg, loop_a) -> Tuple[float, float, float]:
  fitness = 1

  start_a = pm4py.get_start_activities(loop_a)
  end_a = pm4py.get_end_activities(loop_a)

  total_start_activities = 0
  for a in start_a:
    total_start_activities += start_a[a]

  base_probabilities = {}
  actual_probabilities = {}

  for end_activity in end_a:
    total_transitions = 0
    for start_activity in start_a:
      total_transitions += dfg[(end_activity, start_activity)]
      actual_probabilities[end_activity, start_activity] = dfg[(end_activity, start_activity)]
      base_probabilities[end_activity, start_activity] = start_a[start_activity] / total_start_activities

    if total_transitions != 0:
      for start_activity in start_a:
        actual_probabilities[end_activity, start_activity] /= total_transitions
    else:
      for start_activity in start_a:
        actual_probabilities[end_activity, start_activity] = 0

  precision = 1 - calculate_mae(
    [base_probabilities[a,b] for a in end_a for b in start_a], 
    [actual_probabilities[a,b] for a in end_a for b in start_a]
  )

  equality_term = len(loop_a) *len([]) / ((0.5*len(loop_a)+len([]))**2)
  return fitness, precision, f1_score(precision, fitness) * CUSTOM_WEIGHTS["tau_loop"]*equality_term
