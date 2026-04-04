from typing import Tuple
from ..util import get_log_statistics
from .cut_quality import evalutate_cut
from .cut_detection import findCut_OptIMIIst_without_filters, findCut_OptIMIIst_with_filters
from pm4py.objects.process_tree.obj import ProcessTree, Operator
from ..split_log import split_log

def get_optimiist_cut(log, empty_traces=0, filter=True) -> ProcessTree:
  log_stats = get_log_statistics(log)

  if filter:
    C = findCut_OptIMIIst_with_filters(log, log_stats)
  else:
    C = findCut_OptIMIIst_without_filters(log_stats["dfg"], log_stats["efg"], log_stats["start_activities"], log_stats["end_activities"], log_stats["activities"])

  C.append((Operator.LOOP, log_stats["activities"], []))

  operator, L_1, L_2, empty_traces_1, empty_traces_2 = None, None, None, None, None
  a_res = 0
  for cut in C:
    log_a, log_b, empty_traces_a, empty_traces_b = split_log(log, cut[0], cut[1], cut[2], empty_traces, *(cut[3:]) if filter else [])
    a = evalutate_cut(cut, log, log_a, log_b, log_stats["dfg"],log_stats["efg"])[2] - ((len(cut[3]) / len(log_stats["activities"])) if filter and len(cut) > 3 else 0)
    if a_res < a:
      a_res = a
      operator, L_1, L_2, empty_traces_1, empty_traces_2 = cut[0], log_a, log_b, empty_traces_a, empty_traces_b
  
  return operator, L_1, L_2, empty_traces_1, empty_traces_2
