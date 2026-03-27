from .find_cuts_without_filters import sequence_cut_base_model, xor_cut_base_model, parralel_cut_base_model, loop_cut_base_model
from .utils import build_skip_dfg, extract_partitions_pulp, extract_filtered_activity_pulp, get_solver
from ..cut_quality import evalutate_cut
from ...split_log import split_log
from ...util import get_log_statistics
from pm4py.objects.process_tree.obj import Operator
import math
from pulp import *

solver = get_solver()

MAX_FILTERS = math.inf

def add_partition_constraints(ilp_model, x, n, activities):
  # Partitions can not be empty
  ilp_model += lpSum(x[activity] - n[activity] for activity in activities) >= 1
  ilp_model += lpSum(1 - x[activity] for activity in activities) >= 1
  # N is only positive a single time
  ilp_model += lpSum(n[activity] for activity in activities) == 1
  # Positive N is in the first partition
  for activity in activities:
    ilp_model += n[activity] <= x[activity]

def findCut_OptIMIIst(log, log_stats) -> tuple[Operator, list, list, list]:
  seq_partitions, filtered_seq = get_filtered_sequence_cut(log, log_stats)
  xor_paritions, filtered_xor = get_filtered_xor_cut(log, log_stats)
  par_partitions, filtered_par = get_filtered_parallel_cut(log, log_stats)
  loop_partitions, fitlered_loop = get_filtered_loop_cut(log, log_stats)

  return [(Operator.SEQUENCE, seq_partitions[0], seq_partitions[1], filtered_seq), 
          (Operator.XOR, xor_paritions[0], xor_paritions[1], filtered_xor), 
          (Operator.PARALLEL, par_partitions[0], par_partitions[1], filtered_par), 
          (Operator.LOOP, loop_partitions[0], loop_partitions[1], fitlered_loop)]

def get_filtered_cut_iterative(log, operator: Operator, base_model: callable, filter_model: callable, log_stats):
  activities = log["concept:name"].unique()
  total_activities = len(log)

  n_res = []

  if operator == Operator.PARALLEL or operator == Operator.LOOP:
    cut = base_model(log_stats["dfg"], activities, log_stats["start_activities"], log_stats["end_activities"])
  elif operator == Operator.SEQUENCE:
    cut = base_model(log_stats["efg"], activities)
  else:
    cut = base_model(log_stats["dfg"], activities)

  cut = [operator, cut[0][0], cut[0][1]]
  log_a, log_b, empty_traces_a, empty_traces_b = split_log(log, operator, cut[1], cut[2], 0)

  base_score = evalutate_cut(cut, log, log_a, log_b, log_stats["dfg"])[2]

  nsum = 0

  for i in range(min((len(activities) - 2), MAX_FILTERS)):
    if operator == Operator.SEQUENCE:
      cut2 = filter_model(log_stats["efg"], activities)
    else:
      cut2 = filter_model(log, activities)
    n = cut2[1]
    nsum += len(log.loc[log['concept:name'] == n])
    cut2 = [operator, cut2[0][0], cut2[0][1]]
    # Remove the activity from the log
    log = log.loc[log['concept:name'] != n]
    log_stats = get_log_statistics(log)
    activities = log["concept:name"].unique()

    log_a, log_b, empty_traces_a, empty_traces_b = split_log(log, cut2[0], cut2[1], cut2[2], 0)

    score = evalutate_cut(cut2, log, log_a, log_b, log_stats["dfg"])[2] * (1 - (nsum / total_activities))

    if score > base_score:
      n_res.append(n)
      base_score = score
      cut = cut2
    else:
      break
    
  # Remove the n_res activities from the cut
  for n in n_res:
    if n in cut[1]:
      cut[1].remove(n)

  return (cut[1], cut[2]), n_res

def get_filtered_sequence_cut(log, log_stats):
  return get_filtered_cut_iterative(log, Operator.SEQUENCE, sequence_cut_base_model, sequence_cut_filter_model, log_stats)

def get_filtered_xor_cut(log, log_stats):
  return get_filtered_cut_iterative(log, Operator.XOR, xor_cut_base_model, xor_cut_filter_model, log_stats)

def get_filtered_parallel_cut(log, log_stats):
  return get_filtered_cut_iterative(log, Operator.PARALLEL, parralel_cut_base_model, parralel_cut_filter_model, log_stats)

def get_filtered_loop_cut(log, log_stats):
  return get_filtered_cut_iterative(log, Operator.LOOP, loop_cut_base_model, loop_cut_filter_model, log_stats)

def sequence_cut_filter_model(efg, activities, verb=0):
  ilp_model = LpProblem("Sequence_Cut", LpMaximize)

  # Variable X: Binary variable for each activity
  # X = 1 if activity is in the first partition
  # X = 0 if activity is in the second partition
  x = {}

  # Variable n: Binary variable for each activity
  # n = 1 activity is filtered
  n = {}

  # Variable X_abs for each pair of activities
  # x_abs = 1 if a1 is 1 and a1 and a2 are not filtered
  # z = 0 if a1 is 0 or a1 or a2 are filtered
  x_abs = {}

  for activity1 in activities:
    x[activity1] = LpVariable(f"x_{activity1}", cat=LpBinary)
    n[activity1] = LpVariable(f"n_{activity1}", cat=LpBinary)
    for activity2 in activities:
      x_abs[activity1, activity2] = LpVariable(f"z_{activity1}_{activity2}", cat=LpBinary)

  add_partition_constraints(ilp_model, x, n, activities)

  # Only one activity can be filtered 
  ilp_model += lpSum(n[activity] for activity in activities) == 1

  for a in activities:
    for b in activities:
      ilp_model += x_abs[a, b] <= x[a]
      ilp_model += x_abs[a, b] <= 1 - n[a]
      ilp_model += x_abs[a, b] <= 1 - n[b]
      ilp_model += x_abs[a, b] >= x[a] + (1 - n[a]) + (1 - n[b]) - 2

  ilp_model += lpSum((x_abs[a, b] - x_abs[b, a]) * efg[(a, b)] for a in activities for b in activities)

  ilp_model.solve(solver(msg=verb))

  return extract_partitions_pulp(x), extract_filtered_activity_pulp(n), value(ilp_model.objective)

def xor_cut_filter_model(log, activities, verb=0):
  dfg, skips, _, _ = build_skip_dfg(log, activities)

  ilp_model = LpProblem("XOR_Cut", LpMinimize)

  # Variable X: Binary variable for each activity
  # X = 1 if activity is in the first partition
  # X = 0 if activity is in the second partition
  x = {}
  # Variable z: Binary variable for each pair of activities
  # z = 1 if activities are in different partition
  # z = 0 if both activities are in the same partition
  z = {}
  # Variable n: Binary variable for each activity
  # n = 1 activity is filtered
  n = {}

  # Variables k: Binary variable for each tripel of activities
  # k = 1 if the pair of activities is in the same partition
  k = {}

  for activity in activities:
    x[activity] = LpVariable(f"x_{activity}", cat=LpBinary)
    n[activity] = LpVariable(f"n_{activity}", cat=LpBinary)
    for activity2 in activities:
      z[(activity, activity2)] = LpVariable(f"z_{activity}_{activity2}", cat=LpBinary)
      for activity3 in activities:
        k[(activity, activity2, activity3)] = LpVariable(f"k_{activity}_{activity2}_{activity3}", cat=LpBinary)

  add_partition_constraints(ilp_model, x, n, activities)

  # Link values of x to the z variables
  for a in activities:
    for b in activities:
      ilp_model += x[a] - x[b] <= z[(a, b)] + n[a] + n[b]
      ilp_model += x[b] - x[a] <= z[(a, b)] + n[a] + n[b]
      ilp_model += z[(a, b)] <= 1 - n[a]
      ilp_model += z[(a, b)] <= 1 - n[b]

  # Link values of x to the k variables
  for a in activities:
    for b in activities:
      for c in activities:
        # k is zero if n[b] is 0
        ilp_model += k[(a, b, c)] <= n[b]

        ilp_model += k[(a, b, c)] >= x[a] - x[c] - (1 - n[b])
        ilp_model += k[(a, b, c)] >= x[c] - x[a] - (1 - n[b])
        ilp_model += k[(a, b, c)] <= x[a] + x[c]
        ilp_model += k[(a, b, c)] <= 2 - x[a] - x[c]

  # Objective: Minimize the number of arcs between the partitions
  ilp_model += (
    lpSum(dfg[(a, b)] * z[(a, b)] for a in activities for b in activities)
    + lpSum(skips[(b, (a, c))] * k[(a, b, c)] for a in activities for b in activities for c in activities)
  )

  ilp_model.solve(solver(msg=verb))

  return extract_partitions_pulp(x), extract_filtered_activity_pulp(n), value(ilp_model.objective)

def parralel_cut_filter_model(log, activities, verb=0):
  dfg, skips, start_activities_skips, end_activities_skips = build_skip_dfg(log, activities)

  ilp_model = LpProblem("Parallel_Cut", LpMaximize)

  # Variable X: Binary variable for each activity
  # X = 1 if activity is in the first partition
  # X = 0 if activity is in the second partition
  x = {}
  # Variable Z: Binary variable for each pair of activities
  # Z = 1 if activities are in different partition
  # Z = 0 if both activities are in the same partition
  z = {}
  # Variable N: Binary variable for each activity
  # N = 1 activity is filtered
  n = {}
  # Variable K: Binary variable for each pair of activities
  # K = 1 if activities are in different partition
  k = {}

  for activity in activities:
    n[activity] = LpVariable(f"n_{activity}", cat=LpBinary)
    x[activity] = LpVariable(f"x_{activity}", cat=LpBinary)
    for activity2 in activities:
      z[(activity, activity2)] = LpVariable(f"z_{activity}_{activity2}", cat=LpBinary)
      for activity3 in activities:
        k[(activity, activity2, activity3)] = LpVariable(f"k_{activity}_{activity2}_{activity3}", cat=LpBinary)

  # Variable B: Integer variable for the difference between the start and end activities between the partitions
  b = LpVariable("b", 0, None, LpInteger)
  # Helper Variables B_s and B_e: Integer variables for the difference between the start and end activities between the partitions
  b_s = LpVariable("b_s", 0, None, LpInteger)
  b_e = LpVariable("b_e", 0, None, LpInteger)

  add_partition_constraints(ilp_model, x, n, activities)

  # Link values of x to the z variables
  for activity_1 in activities:
    for activity_2 in activities:
      ilp_model += x[activity_1] - x[activity_2] <= z[(activity_1, activity_2)] + n[activity_1] + n[activity_2]
      ilp_model += x[activity_2] - x[activity_1] <= z[(activity_1, activity_2)] + n[activity_1] + n[activity_2]
      ilp_model += z[(activity_1, activity_2)] <= x[activity_1] + x[activity_2]
      ilp_model += z[(activity_1, activity_2)] <= 2 - x[activity_1] - x[activity_2]
      ilp_model += z[(activity_1, activity_2)] <= 1 - n[activity_1]
      ilp_model += z[(activity_1, activity_2)] <= 1 - n[activity_2]

  # Link values of x to the k variables
  for activity_1 in activities:
    for activity_2 in activities:
      for activity_3 in activities:
        # k is zero if n[b] is 0
        ilp_model += k[(activity_1, activity_2, activity_3)] <= n[activity_2]
        
        ilp_model += k[(activity_1, activity_2, activity_3)] >= x[activity_1] - x[activity_3] - (1 - n[activity_2])
        ilp_model += k[(activity_1, activity_2, activity_3)] >= x[activity_3] - x[activity_1] - (1 - n[activity_2])
        ilp_model += k[(activity_1, activity_2, activity_3)] <= x[activity_1] + x[activity_3]
        ilp_model += k[(activity_1, activity_2, activity_3)] <= 2 - x[activity_1] - x[activity_3]

  # Variable xor_n: Binary variable for each pair of activities
  l = {}
  for activity_1 in activities:
    for activity_2 in activities:
      l[activity_1, activity_2] = LpVariable(f"l_{activity_1}_{activity_2}", cat=LpBinary)

  for activity_1 in activities:
      for activity_2 in activities:
          ilp_model += l[activity_1, activity_2] >= x[activity_1] - (1 - n[activity_2])
          ilp_model += l[activity_1, activity_2] <= n[activity_2]
          ilp_model += l[activity_1, activity_2] <= x[activity_1]
  
  # Variable B_s: is the absolute difference between the start activities between the partitions

  ilp_model += b_s >= (lpSum(n[activity_2] * (start_activities_skips[activity_2][activity_1] if activity_1 in start_activities_skips[activity_2] else 0)  for activity_1 in activities for activity_2 in activities) 
    - 2 * lpSum(l[activity_1, activity_2] * (start_activities_skips[activity_2][activity_1] if activity_1 in start_activities_skips[activity_2] else 0) for activity_1 in activities for activity_2 in activities))
  ilp_model += b_s >= ((-1) * lpSum(n[activity_2] * (start_activities_skips[activity_2][activity_1] if activity_1 in start_activities_skips[activity_2] else 0)  for activity_1 in activities for activity_2 in activities)
    - 2 * lpSum(l[activity_1, activity_2] * (start_activities_skips[activity_2][activity_1] if activity_1 in start_activities_skips[activity_2] else 0)  for activity_1 in activities for activity_2 in activities))

  # Variable B_e: is the absolute difference between the end activities between the partitions
  ilp_model += b_e >= (lpSum(n[activity_2] * (end_activities_skips[activity_2][activity_1] if activity_1 in end_activities_skips[activity_2] else 0) for activity_1 in activities for activity_2 in activities)
    - 2 * lpSum(l[activity_1, activity_2] * (end_activities_skips[activity_2][activity_1] if activity_1 in end_activities_skips[activity_2] else 0) for activity_1 in activities for activity_2 in activities))
  ilp_model += b_e >= ((-1) * lpSum(n[activity_2] * (end_activities_skips[activity_2][activity_1] if activity_1 in end_activities_skips[activity_2] else 0) for activity_1 in activities for activity_2 in activities)
    - 2 * lpSum(l[activity_1, activity_2] * (end_activities_skips[activity_2][activity_1] if activity_1 in end_activities_skips[activity_2] else 0) for activity_1 in activities for activity_2 in activities))

  # Variable B: is the sum of the difference between the start and end activities between the partitions
  ilp_model += b >= b_s + b_e

  # Objective: Maximize the number of arcs between the partitions - reduce objective by the difference between start and end activities between the partitions
  ilp_model += (lpSum(dfg[(activity1, activity2)] * z[(activity1, activity2)] for activity1 in activities for activity2 in activities) 
            + lpSum(skips[(activity2, (activity1, activity3))] * k[(activity1, activity2, activity3)] for activity1 in activities for activity2 in activities for activity3 in activities) 
            - b)

  ilp_model.solve(solver(msg=verb))

  return extract_partitions_pulp(x), extract_filtered_activity_pulp(n), value(ilp_model.objective)

def loop_cut_filter_model(log, activities, verb=0):

  dfg, skip_dfgs, start_activities_skips, end_activities_skips = build_skip_dfg(log, activities)

  ilp_model = LpProblem("Loop_Cut", LpMaximize)

  # Variable X: Binary variable for each activity
  # X = 1 if activity is in the first partition
  # X = 0 if activity is in the second partition
  x = {}
  # Variable N: Binary variable for each activity
  # N = 1 activity is filtered
  n = {}

  # Variable W: Binary variable for each pair of activities
  # is 1 if the first activity is an end activity and the second activity is a start activity of the redo loop
  w = {}
  # Variable W_K: Binary variable for each tripel of activities
  # is 1 if w_A_C would be 1 but b is filtered
  w_k = {}

  # Variable V: Binary variable for each pair of activities
  # is 1 if the first activity is an end activity of the redo loop the second activity is a start activity
  v = {}
  # Variable V_K: Binary variable for each tripel of activities
  # is 1 if v_A_C would be 1 but b is filtered
  v_k = {}

  # Variable F: Binary variable for each pair of activities
  # Is the flow between the partitions breaking the loop behaviour
  f = {}
  # Variable F_K: Binary variable for each tripel of activities
  # Is the flow between the partitions breaking the loop behaviour
  f_k = {}

  for activity_1 in activities:
    x[activity_1] = LpVariable(f"x_{activity_1}", cat=LpBinary)
    n[activity_1] = LpVariable(f"n_{activity_1}", cat=LpBinary)
    for activity_2 in activities:
      w[activity_1, activity_2] = LpVariable(f"w_{activity_1}_{activity_2}", cat=LpBinary)
      v[activity_1, activity_2] = LpVariable(f"v_{activity_1}_{activity_2}", cat=LpBinary)
      f[activity_1, activity_2] = LpVariable(f"f_{activity_1}_{activity_2}", cat=LpBinary)
      for activity_3 in activities:
        w_k[activity_1, activity_2, activity_3] = LpVariable(f"w_k_{activity_1}_{activity_2}_{activity_3}", cat=LpBinary)
        v_k[activity_1, activity_2, activity_3] = LpVariable(f"v_k_{activity_1}_{activity_2}_{activity_3}", cat=LpBinary)
        f_k[activity_1, activity_2, activity_3] = LpVariable(f"f_k_{activity_1}_{activity_2}_{activity_3}", cat=LpBinary)

  add_partition_constraints(ilp_model, x, n, activities)

  # Link values of x to the w and y variables
  for activity_1 in activities:
    for activity_2 in activities:
        # Arc is loop conforming if the first activity is an end activity and the second activity is a start activity of the redo loop
        ilp_model += w[activity_1, activity_2] >= (1 - x[activity_2]) + lpSum((1 if activity_1 in end_activities_skips[activity_3] else 0) * n[activity_3] for activity_3 in activities) - 1
        # w_a_b is 0 if b is not a start activity redo
        ilp_model += w[activity_1, activity_2] <= (1 - x[activity_2])
        # w_a_b is 0 if a is not an end activity redo
        ilp_model += w[activity_1, activity_2] <= lpSum((1 if activity_1 in end_activities_skips[activity_3] else 0) * n[activity_3] for activity_3 in activities)

        # Arc is loop conforming if the first activity is an end activity of the redo loop the second activity is a start activity
        ilp_model += v[activity_1, activity_2] >= (1 - x[activity_1]) + lpSum((1 if activity_2 in start_activities_skips[activity_3] else 0) * n[activity_3] for activity_3 in activities) - 1
        # y_a_b is 0 if a is not a redo end activity
        ilp_model += v[activity_1, activity_2] <= (1 - x[activity_1])
        # y_a_b is 0 if b is not a start activity redo
        ilp_model += v[activity_1, activity_2] <= lpSum((1 if activity_2 in start_activities_skips[activity_3] else 0) * n[activity_3] for activity_3 in activities)

        for activity_3 in activities:
          ilp_model += w_k[activity_1, activity_2, activity_3] <= n[activity_2]
          ilp_model += v_k[activity_1, activity_2, activity_3] <= n[activity_2]
          
          ilp_model += w_k[activity_1, activity_2, activity_3] <= w[activity_1, activity_2]
          ilp_model += v_k[activity_1, activity_2, activity_3] <= v[activity_1, activity_2]

        for activity_3 in activities:
          # Identify Arcs that break the loop behaviour (f_a_b = 1)
          ilp_model += f[(activity_1, activity_2)] >= (x[activity_1]-x[activity_2]) - w[activity_1,activity_2] - v[activity_1,activity_2] - w_k[activity_1,activity_2,activity_2] - v_k[activity_1,activity_2,activity_2]
          ilp_model += f[(activity_1, activity_2)] >= (x[activity_2]-x[activity_1]) - w[activity_1,activity_2] - v[activity_1,activity_2] - w_k[activity_1,activity_2,activity_2] - v_k[activity_1,activity_2,activity_2]
        for activity_3 in activities:
          ilp_model += f_k[(activity_1, activity_2, activity_3)] <= n[activity_2]
          ilp_model += f_k[(activity_1, activity_2, activity_3)] >= f[(activity_1, activity_3)] - (1 - n[activity_2])
        

  l = {}
  for activity_1 in activities:
    for activity_2 in activities:
      l[activity_1, activity_2] = LpVariable(f"l_{activity_1}_{activity_2}", cat=LpBinary)

  for activity_1 in activities:
      for activity_2 in activities:
          ilp_model += l[activity_1, activity_2] >= n[activity_2] - x[activity_1]
          ilp_model += l[activity_1, activity_2] <= n[activity_2]
          ilp_model += l[activity_1, activity_2] <= 1 - x[activity_1]

  # Objective
  ilp_model += (
    lpSum((v[activity_1, activity_2] * dfg[activity_1, activity_2]
           + lpSum(v_k[activity_1, activity_3, activity_2] * skip_dfgs[activity_3, (activity_1, activity_2)] for activity_3 in activities)
           + w[activity_1, activity_2] * dfg[activity_1, activity_2]
           + lpSum(w_k[activity_1, activity_3, activity_2] * skip_dfgs[activity_3, (activity_1, activity_2)] for activity_3 in activities))
           for activity_1 in activities for activity_2 in activities)
    - lpSum(((l[activity_1, activity_2]) * (start_activities_skips[activity_2][activity_1] if activity_1 in start_activities_skips[activity_2] else 0) if activity_1 != activity_2 else 0) for activity_1 in activities for activity_2 in activities)
    - lpSum(((l[activity_1, activity_2]) * (end_activities_skips[activity_2][activity_1] if activity_1 in end_activities_skips[activity_2] else 0) if activity_1 != activity_2 else 0) for activity_1 in activities for activity_2 in activities)
    - lpSum((f[activity_1, activity_2] * dfg[activity_1, activity_2]) for activity_1 in activities for activity_2 in activities)
    - lpSum(f_k[activity_1, activity_3, activity_2] * skip_dfgs[activity_3, (activity_1, activity_2)] for activity_1 in activities for activity_2 in activities for activity_3 in activities)
  )

  ilp_model.solve(solver(msg=verb))

  return extract_partitions_pulp(x), extract_filtered_activity_pulp(n), value(ilp_model.objective)