import copy
from typing import List
import pandas as pd
from pm4py.objects.process_tree.obj import Operator
from optimiist.split_log.split_base_operator import split_base_operator
from optimiist.split_log.split_loop import split_loop
from optimiist.split_log.split_tau_loop import split_tau_loop

def split_log_old(log: pd.DataFrame, operator: Operator, activities_partition_1: List[str], activities_partition_2: List[str], empty_cases: int, filtered_activities: List[str] = []) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """
    Split the log into two partitions based on the activities and the operator
    :param log: the log to be split
    :param activities_partition_1: activities to be included in the first partition
    :param activities_partition_2: activities to be included in the second partition
    :return: the two partitions and the number of empty traces in each partition
    """
    # Remove filtered activities
    log = log[~log["concept:name"].isin(filtered_activities)]
    empty_cases += log["case:concept:name"].nunique() - log[~log["concept:name"].isin(filtered_activities)]["case:concept:name"].nunique()

    # TauSkip
    if operator == Operator.XOR and activities_partition_2 == []:
        return log, pd.DataFrame(), 0, 0

    # TauLoop
    if operator == Operator.LOOP and activities_partition_2 == []:
        return split_tau_loop(log, empty_cases)

    # Loop
    if operator == Operator.LOOP and activities_partition_2 != []:
        return split_loop(log, activities_partition_1, activities_partition_2, empty_cases)
    
    # XOR, Sequence, Parallel
    log_partition_1, log_partition_2, empty_cases_1, empty_cases_2 = split_base_operator(log, activities_partition_1, activities_partition_2, empty_cases)

    # We only put the empty cases in one XOR partition in hope they are handled there
    if operator == Operator.XOR:
        empty_cases_1 = empty_cases
        empty_cases_2 = 0

    return log_partition_1, log_partition_2, empty_cases_1, empty_cases_2




def split_log(
    log: pd.DataFrame,
    operator: Operator,
    activities_partition_1: List[str],
    activities_partition_2: List[str],
    empty_cases: int,
    filtered_activities: List[str] = [],
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:

    # --- Filter activities ---
    log = copy.deepcopy(log)
    log = log[~log["concept:name"].isin(filtered_activities)]
    empty_cases += log["case:concept:name"].nunique() - log[~log["concept:name"].isin(filtered_activities)]["case:concept:name"].nunique()

    # --- TauSkip ---
    if operator == Operator.XOR and activities_partition_2 == []:
        return log, pd.DataFrame(), 0, 0

    # --- TauLoop ---
    if operator == Operator.LOOP and activities_partition_2 == []:
        return split_tau_loop(log, empty_cases)

    # --- Convert DataFrame -> list-of-traces for the core logic ---
    A = set(activities_partition_1)
    B = set(activities_partition_2)

    case_col = "case:concept:name"
    act_col = "concept:name"

    grouped = log.groupby(case_col, sort=False)
    case_ids = list(grouped.groups.keys())
    traces = [list(grouped.get_group(cid)[act_col]) for cid in case_ids]

    # --- Core split logic ---
    if operator == Operator.XOR:
        L1, L2 = _split_xor(traces, A, B)
    elif operator == Operator.SEQUENCE:
        L1, L2 = _split_sequence(traces, A, B)
    elif operator == Operator.PARALLEL:
        L1, L2 = _split_parallel(traces, A, B)
    elif operator == Operator.LOOP:
        L1, L2 = _split_loop(traces, A, B)
    else:
        raise ValueError("Unsupported operator")

    # --- Convert results back to DataFrames ---
    def build_df(case_ids, split_traces):
        rows = []
        for cid, trace in zip(case_ids, split_traces):
            if not trace:
                continue
            case_rows = grouped.get_group(cid).copy()
            remaining = list(trace)
            for _, row in case_rows.iterrows():
                if remaining and row[act_col] == remaining[0]:
                    rows.append(row)
                    remaining.pop(0)
        if rows:
            result = pd.DataFrame(rows)
            result = result.astype(log.dtypes.to_dict())
            return result
        return pd.DataFrame(columns=log.columns).astype(log.dtypes.to_dict())
    log_partition_1 = build_df(case_ids, L1)
    log_partition_2 = build_df(case_ids, L2)

    empty_cases_1 = sum(1 for t in L1 if len(t) == 0)
    empty_cases_2 = sum(1 for t in L2 if len(t) == 0)

    if operator == Operator.XOR:
        empty_cases_1 = empty_cases
        empty_cases_2 = 0

    return log_partition_1, log_partition_2, empty_cases_1, empty_cases_2


# ----------------------------------------------------------
# XOR SPLIT
# ----------------------------------------------------------

def _split_xor(traces, A, B):
    L1, L2 = [], []
    for trace in traces:
        tA = [a for a in trace if a in A]
        tB = [a for a in trace if a in B]
        if tA and not tB:
            L1.append(tA)
            L2.append([])
        elif tB and not tA:
            L1.append([])
            L2.append(tB)
        else:
            if len(tA) >= len(tB):
                L1.append(tA)
                L2.append([])
            else:
                L1.append([])
                L2.append(tB)
    return L1, L2


# ----------------------------------------------------------
# SEQUENCE SPLIT
# ----------------------------------------------------------

def _split_sequence(traces, A, B):
    L1, L2 = [], []
    for trace in traces:
        t1, t2 = _split_trace(trace, A, B)
        L1.append(t1)
        L2.append(t2)
    return L1, L2


def _split_trace(trace: list[str], A: set[str], B: set[str]) -> tuple[list[str], list[str]]:
    relevant = [(idx, e) for idx, e in enumerate(trace) if e in A or e in B]
    best_kept = -1
    best_t1, best_t2 = [], []
    for i in range(len(relevant) + 1):
        left = [e for _, e in relevant[:i]]
        right = [e for _, e in relevant[i:]]
        t1 = [e for e in left if e in A]
        t2 = [e for e in right if e in B]
        kept = len(t1) + len(t2)
        if kept > best_kept:
            best_kept = kept
            best_t1, best_t2 = t1, t2
    return best_t1, best_t2


# ----------------------------------------------------------
# PARALLEL SPLIT
# ----------------------------------------------------------

def _split_parallel(traces, A, B):
    L1, L2 = [], []
    for trace in traces:
        L1.append([x for x in trace if x in A])
        L2.append([x for x in trace if x in B])
    return L1, L2


# ----------------------------------------------------------
# LOOP SPLIT
# ----------------------------------------------------------

def _split_loop(traces, A, B):
    L_body, L_redo = [], []
    for trace in traces:
        if trace and trace[0] not in A:
            L_body.append([])
        if trace and trace[-1] not in A:
            L_body.append([])

        current_segment = []
        mode = "body"

        for act in trace:
            if mode == "body":
                if act in A:
                    current_segment.append(act)
                elif act in B:
                    L_body.append(current_segment)
                    current_segment = [act]
                    mode = "redo"
            else:
                if act in B:
                    current_segment.append(act)
                elif act in A:
                    L_redo.append(current_segment)
                    current_segment = [act]
                    mode = "body"

        if mode == "body":
            L_body.append(current_segment)
        else:
            L_redo.append(current_segment)

    return L_body, L_redo