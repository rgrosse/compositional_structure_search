
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




#UPGRADE_RULES = [('g', ('+', ('*', 'm', 'g'), 'g')),
#                 ('g', ('+', ('*', 'g', 'M'), 'g'))]
PMF_MODEL = ('+', ('*', 'g', 'g'), 'g')
MOG_MODEL = ('+', ('*', 'm', 'g'), 'g')
MOG_TRANSPOSE_MODEL = ('+', ('*', 'g', 'M'), 'g')
IBP_MODEL = ('+', ('*', 'b', 'g'), 'g')
IBP_TRANSPOSE_MODEL = ('+', ('*', 'g', 'B'), 'g')
IRM_MODEL = ('+', ('*', 'm', ('+', ('*', 'g', 'M'), 'g')), 'g')
IRM_TRANSPOSE_MODEL = ('+', ('*', ('+', ('*', 'm', 'g'), 'g'), 'M'), 'g')
BMF_MODEL = ('+', ('*', 'b', ('+', ('*', 'g', 'B'), 'g')), 'g')
CHAIN_MODEL = ('+', ('*', 'c', 'g'), 'g')
CHAIN_TRANSPOSE_MODEL = ('+', ('*', 'g', 'C'), 'g')
KF_MODEL = ('+', ('*', ('+', ('*', 'c', 'g'), 'g'), 'g'), 'g')
#LOTS_OF_GAUSSIANS = ('+', ('*', 'g', ('+', ('*', 'g', 'g'), 'g')), 'g')
SPARSE_CODING_MODEL = ('+', ('*', 's', 'g'), 'g')







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
    #if type(structure) == tuple and structure[-1] != 'g':
    #    return False
    if type(structure) == tuple and structure[0] == 's':
        return False
    #if not ((count_leaf(structure, 'c') <= 1 and count_leaf(structure, 'C') <= 1)):
    #    return False
    return True

def is_factorization(structure):
    structure = collapse_sums(structure)
    if type(structure) == str:
        return True
    if structure[0] == '+' and len(structure) == 3 and structure[-1] == 'g':
        return True
    return False


def list_successors_helper(structure):
    if type(structure) == str:
        return [s for n, f, s in PRODUCTION_RULES if f == structure]
    successors = []
    for pos in range(len(structure)):
        for child_succ in list_successors_helper(structure[pos]):
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

def match_paren(string, first):
    assert string[first] == '('
    i = first
    count = 0
    while True:
        assert i < len(string)
        if string[i] == '(':
            count += 1
        elif string[i] == ')':
            count -= 1
        if count == 0:
            return i
        i += 1
        
    

def get_chunks(string):
    chunks = []
    i = 0
    while i < len(string):
        if string[i] == '(':
            end = match_paren(string, i)
            chunks.append(string[i+1:end])
            i = end + 1
        else:
            chunks.append(string[i])
            i += 1
    return chunks

def split_list(list, split_on):
    result = [[]]
    for elt in list:
        if elt == split_on:
            result.append([])
        else:
            result[-1].append(elt)
    return result


## def parse(string):
##     chunks = get_chunks(string)
##     if len(chunks) == 1:
##         if chunks[0][0] == '(':
##             return parse(chunks[0][1:-1])
##         else:
##             assert len(chunks[0]) == 1
##             return chunks[0]
##     elif '+' in chunks:
##         sublists = split_list(chunks, '+')
##         temp = []
##         for s in sublists:
##             if len(s) == 1:
##                 temp.append(parse(s[0]))
##             else:
##                 parts = tuple([parse(si) for si in s])
##                 temp.append(('*',) + parts)
##         return ('+',) + tuple(temp)
##     else:
##         return ('*',) + tuple([parse(s) for s in chunks])

def parse(string):
    structure = parsing.parse(string)
    return collapse_sums(structure)
    

def find(model, structure):
    result = []
    if model.structure() == structure:
        result.append(structure)
    if not model.isleaf():
        for child in model.children:
            result += find(child, structure)
    return result



        
def generalizes(dist1, dist2):
    if dist1 == dist2: return True
    if dist1 == 'b' and dist2 == 'm': return True
    if dist1 == 'B' and dist2 == 'M': return True
    if dist1 == 'g' and dist2 in ['c', 'C', 'b', 'B', 'm', 'M', 's', 'k']: return True
    if dist1 == 's' and dist2 in ['c', 'C', 'b', 'B', 'm', 'M', 'g', 'k']: return True
    if dist1 == 'k' and dist2 in ['c', 'C', 'b', 'B', 'm', 'M', 'g', 's']: return True

def location_string(node, query):
    if node.isleaf():
        dist = node.distribution()
        if dist.isupper():
            ans = dist.lower() + "'"
        else:
            ans = dist
    elif node.issum():
        parts = [location_string(child, query) for child in node.children]
        ans = ' + '.join(parts)
    elif node.isgsm():
        ans = 's(%s)' % location_string(node.scale_node, query)
    else:
        assert node.isproduct()
        parts = []
        for child in node.children:
            if child.isleaf():
                parts.append(location_string(child, query))
            else:
                parts.append('(' + location_string(child, query) + ')')
        ans = ' '.join(parts)

    if node is query:
        ans = '<' + ans + '>'
    return ans
        
