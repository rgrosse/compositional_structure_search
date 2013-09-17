
import parsing

START = 'g'

PRODUCTION_RULES = {'low-rank':          [('g',     ('+', ('*', 'g', 'g'), 'g'))],
                
                    'clustering':        [('g',     ('+', ('*', 'm', 'g'), 'g')),
                                          ('g',     ('+', ('*', 'g', 'M'), 'g'))],
                    
                    'binary':            [('g',     ('+', ('*', 'b', 'g'), 'g')),
                                          ('g',     ('+', ('*', 'g', 'B'), 'g'))],
                    
                    'chain':             [('g',     ('+', ('*', 'c', 'g'), 'g')),
                                          ('g',     ('+', ('*', 'g', 'C'), 'g'))],
                    
                    'sparsity':          [('g',     ('s', 'g'))],
                    
                    'expand-disc':       [('m',     ('+', ('*', 'm', 'g'), 'g')),
                                          ('M',     ('+', ('*', 'g', 'M'), 'g')),
                                          ('b',     ('+', ('*', 'b', 'g'), 'g')),
                                          ('B',     ('+', ('*', 'g', 'B'), 'g'))],
                    
                    'm-to-b':            [('m',     'b')],
                    }


def is_valid(structure):
    if type(structure) == str and structure != 'g':
        return False
    if type(structure) == tuple and structure[0] == 's':
        return False
    return True

def list_successors_helper(structure, rule_names, is_noise, expand_noise=True):
    rules = reduce(list.__add__, [PRODUCTION_RULES[rn] for rn in rule_names])

    if is_noise and not expand_noise:
        return []
    
    if type(structure) == str:
        return [rhs for lhs, rhs in rules if lhs == structure]
    
    successors = []
    for pos in range(len(structure)):
        is_noise = (structure[0] == '+' and pos == len(structure) - 1)
        for child_succ in list_successors_helper(structure[pos], rule_names, is_noise, expand_noise):
            if is_noise and type(child_succ) == tuple and child_succ[0] == 's':
                continue
            successors.append(structure[:pos] + (child_succ,) + structure[pos+1:])
    return successors

def list_successors(structure, rules, expand_noise=True):
    successors = list_successors_helper(structure, rules, False, expand_noise)
    return filter(is_valid, successors)

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

def list_collapsed_successors(structure, rule_names, expand_noise=True):
    return [collapse_sums(s) for s in list_successors_helper(structure, rule_names, False, expand_noise)
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
    

