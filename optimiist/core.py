from typing import Tuple
import pandas as pd
from pm4py.objects.process_tree.obj import ProcessTree, Operator
from pm4py.objects.process_tree.utils.generic import fold, reduce_tau_leafs
from pm4py.objects.petri_net.obj import Marking
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.bpmn.obj import BPMN
import pm4py

from .base_case import get_base_case
from .inductive_miner_cuts import get_inductive_miner_cuts
from .split_log import split_log,split_log_old
from .optimiist_fallthrough.optimiist_fallthrough import get_optimiist_cut

def optimiist(log: pd.DataFrame, filter: bool = True) -> Tuple[PetriNet, Marking, Marking]:
    return optimiist_pt(log, filter)

def optimiist_pt(log: pd.DataFrame, filter: bool) -> Tuple[PetriNet, Marking, Marking]:
    tree = optimiist_tree(log, filter)
    print(tree)
    petri_net, initial_marking, final_marking = pm4py.convert_to_petri_net(tree)
    return petri_net, initial_marking, final_marking

def optimiist_tree(log: pd.DataFrame, filter: bool) -> ProcessTree:
    tree = optimiist_rec(log, 0, filter)
    tree = fold(reduce_tau_leafs(tree))
    return tree

def optimiist_bpmn(log: pd.DataFrame, filter: bool) -> BPMN:
    tree = optimiist_tree(log, filter)
    bpmn = pm4py.convert_to_bpmn(tree)
    return bpmn

TAU_THRESHOLD = 0.5
LOOP_THRESHOLD = 0.5

def optimiist_rec(log: pd.DataFrame, empty_cases: int = 0, filter: bool = True) -> ProcessTree:
    base_case = get_base_case(log, empty_cases, TAU_THRESHOLD, LOOP_THRESHOLD)
    if base_case is not None:
        return base_case

    if (empty_cases / (log["case:concept:name"].nunique() + empty_cases)) > TAU_THRESHOLD:
        return ProcessTree(operator=Operator.XOR, children=[optimiist_rec(log, 0, filter), ProcessTree()])

    cut = get_inductive_miner_cuts(log)
    if cut is not None:
        log_1, log_2, empty_cases_1, empty_cases_2 = split_log_old(log, cut[0], cut[1], cut[2], empty_cases)
        return ProcessTree(cut[0], children=[optimiist_rec(log_1, empty_cases_1, filter), optimiist_rec(log_2, empty_cases_2, filter)])

    cut = get_optimiist_cut(log, empty_cases, filter)

    return ProcessTree(cut[0], children=[optimiist_rec(cut[1], cut[3], filter), optimiist_rec(cut[2], cut[4], filter)])
