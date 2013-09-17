import collections
import numpy as np
import sys

import grammar




def format_table(table, sep='  '):
    num_cols = len(table[0])
    if any([len(row) != num_cols for row in table]):
        raise RuntimeError('Number of columns must match.')

    widths = [max([len(row[i]) for row in table])
              for i in range(num_cols)]
    format_string = sep.join(['%' + str(w) + 's' for w in widths])
    return [format_string % tuple(row) for row in table]

def format_table_latex(table):
    return [l + ' \\\\' for l in format_table(table, ' & ')]

class Failure:
    def __init__(self, structure, level, all_failed, name=None):
        self.structure = structure
        self.level = level
        self.all_failed = all_failed
        self.name = name

def print_failed_structures(failures, outfile=sys.stdout):
    if failures:
        print >> outfile, 'The inference algorithms failed for the following structures:'
        print >> outfile
        print >> outfile, '%30s%8s        %s' % \
              ('structure', 'level', 'notes')
        print >> outfile
        for f in failures:
            line = '%30s%8d        ' % (grammar.pretty_print(f.structure), f.level)
            if f.name:
                line += '(for %s)  ' % f.name
            if not f.all_failed:
                line += '(only some jobs failed)  '
            print >> outfile, line
        print >> outfile
        print >> outfile


class ModelScore:
    def __init__(self, structure, row_score, col_score, total, row_improvement, col_improvement,
                 z_score_row, z_score_col):
        self.structure = structure
        self.row_score = row_score
        self.col_score = col_score
        self.total = total
        self.row_improvement = row_improvement
        self.col_improvement = col_improvement
        self.z_score_row = z_score_row
        self.z_score_col = z_score_col

def print_scores(level, model_scores, outfile=sys.stdout):
    print >> outfile, 'The following are the top-scoring structures for level %d:' % level
    print >> outfile
    print >> outfile, '%30s%10s%10s%13s%13s%13s%10s%10s' % \
          ('structure', 'row', 'col', 'total', 'row impvt.', 'col impvt.', 'z (row)', 'z (col)')
    print >> outfile
    for ms in model_scores:
        print >> outfile, '%30s%10.2f%10.2f%13.2f%13.2f%13.2f%10.2f%10.2f' % \
              (grammar.pretty_print(ms.structure), ms.row_score, ms.col_score, ms.total,
               ms.row_improvement, ms.col_improvement, ms.z_score_row, ms.z_score_col)
    print >> outfile
    print >> outfile
    

def print_model_sequence(model_scores, outfile=sys.stdout):
    print >> outfile, "Here are the best-performing structures in each level of the search:"
    print >> outfile
    print >> outfile, '%10s%25s%13s%13s%10s%10s' % \
          ('level', 'structure', 'row impvt.', 'col impvt.', 'z (row)', 'z (col)')
    print >> outfile
    for i, ms in enumerate(model_scores):
        print >> outfile, '%10d%25s%13.2f%13.2f%10.2f%10.2f' % \
              (i+1, grammar.pretty_print(ms.structure), ms.row_improvement, ms.col_improvement,
               ms.z_score_row, ms.z_score_col)
    print >> outfile
    print >> outfile


class RunningTime:
    def __init__(self, level, structure, num_samples, total_time):
        self.level = level
        self.structure = structure
        self.num_samples = num_samples
        self.total_time = total_time

def format_time(t):
    if t < 60.:
        return '%1.1f seconds' % t
    elif t < 3600.:
        return '%1.1f minutes' % (t / 60.)
    else:
        return '%1.1f hours' % (t / 3600.)

def print_running_times(running_times, outfile=sys.stdout):
    total = sum([rt.total_time for rt in running_times])
    print >> outfile, 'Total CPU time was %s. Here is the breakdown:' % format_time(total)
    print >> outfile
    print >> outfile, '%30s%8s        %s' % \
          ('structure', 'level', 'time')
    print >> outfile
    running_times = sorted(running_times, key=lambda rt: rt.total_time, reverse=True)
    for rt in running_times:
        time_str = '%d  x  %s' % (rt.num_samples, format_time(rt.total_time / rt.num_samples))
        print >> outfile, '%30s%8d        %s' % (grammar.pretty_print(rt.structure), rt.level, time_str)
    print >> outfile
    print >> outfile


class FinalResult:
    def __init__(self, expt_name, structure):
        self.expt_name = expt_name
        self.structure = structure

def print_learned_structures(results, outfile=sys.stdout):
    def sortkey(result):
        return result.expt_name.split('_')[-1]
    results = sorted(results, key=sortkey)

    print >> outfile, 'The learned structures:'
    print >> outfile
    print >> outfile, '%25s%25s' % ('experiment', 'structure')
    print >> outfile
    for r in results:
        print >> outfile, '%25s%25s' % (r.expt_name, grammar.pretty_print(r.structure))
    print >> outfile
    print >> outfile



class LatentVariables:
    def __init__(self, label, z):
        self.label = label
        self.z = z

def print_components(model, structure, row_or_col, items, outfile=sys.stdout):
    cluster_members = collections.defaultdict(list)
    if model == 'clustering':
        for item in items:
            z = item.z if np.isscalar(item.z) else item.z.argmax()
            cluster_members[z].append(item.label)

        component_type, component_type_pl = 'Cluster', 'clusters'
    elif model == 'binary':
        for item in items:
            for i, zi in enumerate(item.z):
                if zi:
                    cluster_members[i].append(item.label)
        component_type, component_type_pl = 'Component', 'components'
            
    cluster_ids = sorted(cluster_members.keys(), key=lambda k: len(cluster_members[k]), reverse=True)

    row_col_str = {'row': 'row', 'col': 'column'}[row_or_col]
    print >> outfile, 'For structure %s, the following %s %s were found:' % \
          (grammar.pretty_print(structure), row_col_str, component_type_pl)
    print >> outfile

    for i, cid in enumerate(cluster_ids):
        print >> outfile, '    %s %d:' % (component_type, i+1)
        print >> outfile
        for label in cluster_members[cid]:
            print >> outfile, '        %s' % label
        print >> outfile
    print >> outfile


    

