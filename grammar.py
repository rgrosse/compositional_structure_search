
import parsing




START = 'g'

PRODUCTION_RULES = [('low-rank',                    'g',     ('+', ('*', 'g', 'g'), 'g')),
                    ('row-clustering',              'g',     ('+', ('*', 'm', 'g'), 'g')),
                    ('col-clustering',              'g',     ('+', ('*', 'g', 'M'), 'g')),
                    ('row-binary',                  'g',     ('+', ('*', 'b', 'g'), 'g')),
                    ('col-binary',                  'g',     ('+', ('*', 'g', 'B'), 'g')),
                    ('row-chain',                   'g',     ('+', ('*', 'c', 'g'), 'g')),
                    ('col-chain',                   'g',     ('+', ('*', 'g', 'C'), 'g')),
                    ('sparsity',                    'g',     ('s', 'g')),
                    #('row-clustering-to-binary',    'm',     'b'),
                    #('col-clustering-to-binary',    'M',     'B'),
                    #('row-multi-to-clustering',     'm',     ('+', ('*', 'm', 'g'), 'g')),
                    #('col-multi-to-clustering',     'M',     ('+', ('*', 'g', 'M'), 'g')),
                    #('row-integ-to-chain',          'c',     ('+', ('*', 'c', 'g'), 'g')),
                    #('col-integ-to-chain',          'C',     ('+', ('*', 'g', 'C'), 'g')),
                    ]

name2rule = dict((name, (left, right)) for name, left, right in PRODUCTION_RULES)
rule2name = dict(((left, right), name) for name, left, right in PRODUCTION_RULES)

EXPAND_NOISE = True



def count_leaf(structure, leaf):
    if type(structure) == str:
        if structure == leaf:
            return 1
        else:
            return 0
    else:
        return sum([count_leaf(child, leaf) for child in structure])

def is_valid(structure):
    if type(structure) == str and structure != 'g':
        return False
    if type(structure) == tuple and structure[0] == 's':
        return False
    return True

def is_factorization(structure):
    structure = collapse_sums(structure)
    if type(structure) == str:
        return True
    if structure[0] == '+' and len(structure) == 3 and structure[-1] == 'g':
        return True
    return False


def list_successors_helper(structure, is_noise=False):
    if type(structure) == str:
        return [s for n, f, s in PRODUCTION_RULES if f == structure]
    successors = []
    for pos in range(len(structure)):
        is_noise = (structure[0] == '+' and pos == len(structure) - 1)
        for child_succ in list_successors_helper(structure[pos], is_noise):
            if is_noise and type(child_succ) == tuple and child_succ[0] == 's':
                continue
            successors.append(structure[:pos] + (child_succ,) + structure[pos+1:])
    return successors

def list_successors(structure):
    succ = filter(is_valid, list_successors_helper(structure))
    if not EXPAND_NOISE and type(structure) != str:
        succ = filter(is_factorization, succ)
    return succ

def collapse_sums(structure):
    if type(structure) == str:
        return structure
    elif structure[0] == '+':
        new_structure = ('+',)
        for s_ in structure[1:]:
            s = collapse_sums(s_)
            if type(s) == tuple and s[0] == '+':
                new_structure = new_structure + s[1:]
            else:
                new_structure = new_structure + (s,)
        return new_structure
    else:
        return tuple([collapse_sums(s) for s in structure])

def list_collapsed_successors(structure):
    #return [collapse_sums(s) for s in list_successors(structure)]
    return [collapse_sums(s) for s in list_successors_helper(structure)
            if is_valid(collapse_sums(s))]

def pretty_print(structure, spaces=True, quotes=True):
    if spaces:
        PLUS = ' + '
    else:
        PLUS = '+'
    
    if type(structure) == str:
        if structure.isupper() and quotes:
            return structure.lower() + "'"
        else:
            return structure
    elif structure[0] == '+':
        parts = [pretty_print(s, spaces, quotes) for s in structure[1:]]
        return PLUS.join(parts)
    elif structure[0] == 's':
        return 's(%s)' % pretty_print(structure[1], spaces, quotes)
    else:
        assert structure[0] == '*'
        parts = []
        for s in structure[1:]:
            if type(s) == str or s[0] == '*' or s[0] == 's':
                parts.append(pretty_print(s, spaces, quotes))
            else:
                parts.append('(' + pretty_print(s, spaces, quotes) + ')')
        return ''.join(parts)

def list_derivations(depth, do_print=False):
    derivations = [['g']]
    for i in range(depth):
        new_derivations = []
        for d in derivations:
            new_derivations += [d + [s] for s in list_successors(d[-1])]
        derivations = new_derivations

    for d in derivations:
        if do_print:
            print [pretty_print(s) for s in d]

    return derivations

def list_structures(depth):
    full = set()
    for i in range(1, depth+1):
        derivations = list_derivations(depth, False)
        full.update(set([d[-1] for d in derivations]))
    return full




def parse(string):
    structure = parsing.parse(string)
    return collapse_sums(structure)
    

