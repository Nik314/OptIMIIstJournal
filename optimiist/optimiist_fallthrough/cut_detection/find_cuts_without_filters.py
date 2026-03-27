from pm4py.objects.process_tree.obj import Operator
from .utils import extract_partitions_pulp, get_solver
from pulp import *

solver = get_solver()
BASE_ILP_TIMEOUT = 60

def findCut_OptIMIIst(dfg, efg, start_activities, end_activities, activities) -> list[tuple[Operator, list[str], list[str]]]:
  seq = seq_partitions, obj_val = sequence_cut_base_model(efg, activities)
  xor = xor_paritions, obj_val = xor_cut_base_model(dfg, activities)
  par = par_partitions, obj_val = parralel_cut_base_model(dfg, activities, start_activities, end_activities)
  loop = loop_partitions, obj_val = loop_cut_base_model(dfg, activities, start_activities, end_activities)

  return [(Operator.SEQUENCE, seq_partitions[0], seq_partitions[1]), 
          (Operator.XOR, xor_paritions[0], xor_paritions[1]), 
          (Operator.PARALLEL, par_partitions[0], par_partitions[1]), 
          (Operator.LOOP, loop_partitions[0], loop_partitions[1])]

def add_partition_constraints(ilp_model, x, activities):
  # Partitions can not be empty
  ilp_model += lpSum(x[activity] for activity in activities) >= 1
  ilp_model += lpSum(1 - x[activity] for activity in activities) >= 1

def sequence_cut_base_model(efg, activities, verb=0):
  ilp_model = LpProblem("Sequence_Cut", LpMaximize)

  # Variable X: Binary variable for each activity
  # X = 1 if activity is in the first partition
  # X = 0 if activity is in the second partition
  x = LpVariable.dicts("x", activities, 0, 1, LpBinary)

  add_partition_constraints(ilp_model, x, activities)

  # Maximize the arcs from partition 1 to partition 2 - reduce objective by the arcs from partition 2 to partition 1
  ilp_model += lpSum((x[a] - x[b]) * efg[(a, b)] for a in activities for b in activities)

  ilp_model.solve(solver(msg=verb,timeLimit=BASE_ILP_TIMEOUT))

  return extract_partitions_pulp(x), value(ilp_model.objective)

def xor_cut_base_model(dfg, activities, verb=0):
  ilp_model = LpProblem("XOR_Cut", LpMinimize)

  # Variable X: Binary variable for each activity
  # X = 1 if activity is in the first partition
  # X = 0 if activity is in the second partition
  x = LpVariable.dicts("x", activities, 0, 1, LpBinary)
  # Variable z: Binary variable for each pair of activities
  # z = 1 if activities are in different partition
  # z = 0 if both activities are in the same partition
  z = LpVariable.dicts("z", [(a, b) for a in activities for b in activities], 0, 1, LpBinary)

  add_partition_constraints(ilp_model, x, activities)

  # Link values of x to the z variables
  for a in activities:
    for b in activities:
      ilp_model += x[a] - x[b] <= z[(a, b)]
      ilp_model += x[b] - x[a] <= z[(a, b)]

  # Objective: Minimize the number of arcs between the partitions
  ilp_model += lpSum(dfg[(a, b)] * z[(a, b)] for a in activities for b in activities)

  ilp_model.solve(solver(msg=verb,timeLimit=BASE_ILP_TIMEOUT))

  return extract_partitions_pulp(x), value(ilp_model.objective)

def parralel_cut_base_model(dfg, activities, start_activities, end_activities, verb=0):
  ilp_model = LpProblem("Parallel_Cut", LpMaximize)

  # Variable X: Binary variable for each activity
  # X = 1 if activity is in the first partition
  # X = 0 if activity is in the second partition
  x = LpVariable.dicts("x", activities, 0, 1, LpBinary)
  # Variable Z: Binary variable for each pair of activities
  # Z = 1 if activities are in different partition
  # Z = 0 if both activities are in the same partition
  z = LpVariable.dicts("z", [(a, b) for a in activities for b in activities], 0, 1, LpBinary)
  # Variable B: Integer variable for the difference between the start and end activities between the partitions
  b_f = LpVariable("b", 0, None, LpInteger)
  # Helper Variables B_s and B_e: Integer variables for the difference between the start and end activities between the partitions
  b_s = LpVariable("b_s", 0, None, LpInteger)
  b_e = LpVariable("b_e", 0, None, LpInteger)

  add_partition_constraints(ilp_model, x, activities)

  # Link values of x to the z variables
  for a in activities:
    for b in activities:
      ilp_model += z[(a, b)] >= x[a] - x[b]
      ilp_model += z[(a, b)] >= x[b] - x[a]
      ilp_model += z[(a, b)] <= x[a] + x[b]
      ilp_model += z[(a, b)] <= 2 - x[a] - x[b]

  # Variable B_s: is the absolute difference between the start activities between the partitions
  ilp_model += b_s >= lpSum((start_activities[a] if a in start_activities else 0) for a in activities) - 2 * lpSum((x[a] * (start_activities[a] if a in start_activities else 0) for a in activities))
  ilp_model += b_s >= (-1) * lpSum((start_activities[a] if a in start_activities else 0) for a in activities) - 2 * lpSum((x[a] * (start_activities[a] if a in start_activities else 0) for a in activities))

  # Variable B_e: is the absolute difference between the end activities between the partitions
  ilp_model += b_e >= lpSum((end_activities[a] if a in end_activities else 0) for a in activities) - 2 * lpSum((x[a] * (end_activities[a] if a in end_activities else 0) for a in activities))
  ilp_model += b_e >= (-1) * lpSum((end_activities[a] if a in end_activities else 0) for a in activities) - 2 * lpSum((x[a] * (end_activities[a] if a in end_activities else 0) for a in activities))

  # Variable B: is the sum of the difference between the start and end activities between the partitions
  ilp_model += b_f == b_s + b_e

  # Objective: Maximize the number of arcs between the partitions - reduce objective by the difference between start and end activities between the partitions
  ilp_model += lpSum(z[(act1, act2)] * dfg[(act1, act2)] for act1 in activities for act2 in activities) - b_f

  ilp_model.solve(solver(msg=verb,timeLimit=BASE_ILP_TIMEOUT))

  return extract_partitions_pulp(x), value(ilp_model.objective)

def loop_cut_base_model(dfg, activities, start_activities, end_activities, verb=0):
  ilp_model = LpProblem("Loop_Cut", LpMaximize)

  # Variable X: Binary variable for each activity
  # X = 1 if activity is in the first partition
  # X = 0 if activity is in the second partition
  x = LpVariable.dicts("x", activities, 0, 1, LpBinary)

  # Variable W: Binary variable for each pair of activities
  # is 1 if the first activity is an end activity and the second activity is a start activity of the redo loop
  w = LpVariable.dicts("w", [(a, b) for a in activities for b in activities], 0, 1, LpBinary)
  # Variable V: Binary variable for each pair of activities
  # is 1 if the first activity is an end activity of the redo loop the second activity is a start activity
  v = LpVariable.dicts("v", [(a, b) for a in activities for b in activities], 0, 1, LpBinary)
  # Variable F: Binary variable for each pair of activities
  # Is the flow between the partitions breaking the loop behaviour
  f = LpVariable.dicts("f", [(a, b) for a in activities for b in activities], 0, 1, LpBinary)

  add_partition_constraints(ilp_model, x, activities)

  # Link values of x to the w and v variables
  for activity_1 in activities:
    for activity_2 in activities:
      # Arc is loop conforming if the first activity is an end activity and the second activity is a start activity of the redo loop
      ilp_model += w[(activity_1, activity_2)] >= (1 - x[activity_2]) + (1 if activity_1 in end_activities else 0) - 1
      # w_a_b is 0 if b is not a start activity redo
      ilp_model += w[(activity_1, activity_2)] <= (1 - x[activity_2])
      # w_a_b is 0 if a is not an end activity redo
      ilp_model += w[(activity_1, activity_2)] <= (1 if activity_1 in end_activities else 0)

      # Arc is loop conforming if the first activity is an end activity of the redo loop the second activity is a start activity
      ilp_model += v[(activity_1, activity_2)] >= (1 - x[activity_1]) + (1 if activity_2 in start_activities else 0) - 1
      # v_a_b is 0 if a is not a redo end activity
      ilp_model += v[(activity_1, activity_2)] <= (1 - x[activity_1])
      # v_a_b is 0 if b is not a start activity redo
      ilp_model += v[(activity_1, activity_2)] <= (1 if activity_2 in start_activities else 0)

      # Identify Arcs that break the loop behaviour (f_a_b = 1)
      ilp_model += f[(activity_1, activity_2)] >= (x[activity_1] - x[activity_2]) - w[(activity_1, activity_2)] - v[(activity_1, activity_2)]
      ilp_model += f[(activity_1, activity_2)] >= (x[activity_2] - x[activity_1]) - w[(activity_1, activity_2)] - v[(activity_1, activity_2)]

  # Objective
  ilp_model += (
    lpSum((v[(activity_1, activity_2)] * dfg[(activity_1, activity_2)] 
           + w[(activity_1, activity_2)] * dfg[(activity_1, activity_2)])
          for activity_1 in activities for activity_2 in activities) 
    - lpSum((1 - x[a]) * ((start_activities[a] if a in start_activities else 0) + (end_activities[a] if a in end_activities else 0)) for a in activities)
    - lpSum((f[(activity_1, activity_2)] * dfg[(activity_1, activity_2)]) for activity_1 in activities for activity_2 in activities)
  )

  ilp_model.solve(solver(msg=verb,timeLimit=BASE_ILP_TIMEOUT))

  return extract_partitions_pulp(x), value(ilp_model.objective)
