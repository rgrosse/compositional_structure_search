import numpy as np
nax = np.newaxis

import recursive

class Leaf:
    def __init__(self, left_side, right_side, fixed):
        self.left_side = left_side
        self.right_side = right_side
        self.fixed = fixed
        self.children = []

    def structure(self):
        return self.distribution()

    def transpose(self):
        return self.transpose_class()(self.right_side, self.left_side)

    def display(self, indent=0):
        s = self.__class__.__name__
        if self.fixed:
            s += ', fixed'
        s = ' ' * indent + s
        if hasattr(self, 'id'):
            s = '(%2d)  ' % self.id + s
        print s

    def dummy(self):
        return self.node_class().dummy()

class Gaussian(Leaf):
    def __init__(self, variance_type, fixed_variance, left_side, right_side, fixed):
        Leaf.__init__(self, left_side, right_side, fixed)
        if variance_type not in ['row', 'col', 'scalar']:
            raise RuntimeError('Unknown variance type: %s' % variance_type)
        self.variance_type = variance_type
        self.fixed_variance = fixed_variance

    def distribution(self):
        return 'g'

    def transpose_class(self):
        return Gaussian

    def node_class(self):
        return recursive.GaussianNode

    def transpose(self):
        if self.variance_type == 'row':
            variance_type = 'col'
        elif self.variance_type == 'col':
            variance_type = 'row'
        elif self.variance_type == 'scalar':
            variance_type = 'scalar'
        return Gaussian(variance_type, self.fixed_variance, self.right_side, self.left_side)
    
    def display(self, indent=0):
        s = 'Gaussian, %s' % self.variance_type
        if self.fixed_variance:
            s += ', fixed_variance'
        if self.fixed:
            s += ', fixed'
        s = ' ' * indent + s
        if hasattr(self, 'id'):
            s = '(%2d)  ' % self.id + s
        print s

    def dummy(self):
        return recursive.GaussianNode.dummy(self.variance_type)

class Multinomial(Leaf):
    def distribution(self):
        return 'm'

    def transpose_class(self):
        return MultinomialT

    def node_class(self):
        return recursive.MultinomialNode

class MultinomialT(Leaf):
    def distribution(self):
        return 'M'

    def transpose_class(self):
        return Multinomial

    def node_class(self):
        return recursive.MultinomialTNode

class Bernoulli(Leaf):
    def distribution(self):
        return 'b'

    def transpose_class(self):
        return BernoulliT

    def node_class(self):
        return recursive.BernoulliNode

class BernoulliT(Leaf):
    def distribution(self):
        return 'B'

    def transpose_class(self):
        return Bernoulli

    def node_class(self):
        return recursive.BernoulliTNode

class Integration(Leaf):
    def distribution(self):
        return 'c'

    def transpose_class(self):
        return IntegrationT

    def node_class(self):
        return recursive.IntegrationNode



class IntegrationT(Leaf):
    def distribution(self):
        return 'C'

    def transpose_class(self):
        return Integration

    def node_class(self):
        return recursive.IntegrationTNode

class GSM:
    def __init__(self, left_side, right_side, fixed, scale_node, bias_type):
        self.left_side = left_side
        self.right_side = right_side
        self.fixed = fixed
        self.scale_node = scale_node
        if bias_type not in ['row', 'col', 'scalar']:
            raise RuntimeError('Unknown bias type: %s' % bias_type)
        self.bias_type = bias_type
        self.children = [self.scale_node]

    def structure(self):
        return ('s', self.scale_node.structure())

    def transpose(self):
        return GSM(self.scale_node.transpose())

    def display(self, indent=0):
        s = ' ' * indent + 'GSM'
        if hasattr(self, 'id'):
            s = '(%2d)  ' % self.id + s
        print s

        self.scale_node.display(indent + 4)

    def dummy(self):
        value = np.zeros((5, 5))
        if self.bias_type in ['row', 'col']:
            bias = np.zeros(5)
        else:
            bias = 0.
        return recursive.GSMNode(value, self.scale_node.dummy(), self.bias_type, bias)


class Sum:
    def __init__(self, children, left_side, right_side, fixed):
        self.children = children
        self.left_side = left_side
        self.right_side = right_side
        self.fixed = fixed

    def structure(self):
        return ('+',) + tuple([c.structure() for c in self.children])

    def transpose(self):
        return Sum([c.transpose() for c in self.children], self.right_side, self.left_side)

    def display(self, indent=0):
        s = ' ' * indent + 'Sum'
        if hasattr(self, 'id'):
            s = '(%2d)  ' % self.id + s
        print s

        for c in self.children:
            c.display(indent + 4)

    def dummy(self):
        return recursive.SumNode([c.dummy() for c in self.children])
    

class Product:
    def __init__(self, left, right, left_side, right_side, fixed):
        self.left = left
        self.right = right
        self.children = [left, right]
        self.left_side = left_side
        self.right_side = right_side
        self.fixed = fixed

    def structure(self):
        return ('*',) + tuple([self.left.structure(), self.right.structure()])

    def transpose(self):
        return Product(self.right.transpose(), self.left.transpose(), self.obs.T.copy())

    def display(self, indent=0):
        s = ' ' * indent + 'Product'
        if hasattr(self, 'id'):
            s = '(%2d)  ' % self.id + s
        print s
        
        for c in [self.left, self.right]:
            c.display(indent + 4)

    def dummy(self):
        return recursive.ProductNode([self.left.dummy(), self.right.dummy()])


def continuous_left(structure):
    if type(structure) == str:
        return structure in ['g', 's', 'k']
    elif type(structure) == tuple and structure[0] == '+':
        return any([continuous_left(c) for c in structure[1:]])
    elif type(structure) == tuple and structure[0] == '*':
        assert len(structure) == 3
        return continuous_left(structure[1])
    elif type(structure) == tuple and structure[0] == 's':
        return True
    else:
        raise RuntimeError('Invalid structure: %s' % structure)

def continuous_right(structure):
    if type(structure) == str:
        return structure == 'g'
    elif type(structure) == tuple and structure[0] == '+':
        return any([continuous_right(c) for c in structure[1:]])
    elif type(structure) == tuple and structure[0] == '*':
        assert len(structure) == 3
        return continuous_right(structure[2])
    elif type(structure) == tuple and structure[0] == 's':
        return True
    else:
        raise RuntimeError('Invalid structure: %s' % str(structure))
    
    


dist2class = {'g': Gaussian,
              'm': Multinomial,
              'M': MultinomialT,
              'b': Bernoulli,
              'B': BernoulliT,
              'c': Integration,
              'C': IntegrationT,
              }

def get_model_helper(structure, left_side, right_side, fixed, fixed_variance, variance_type):
    if type(structure) == str:
        if structure == 'g':
            return Gaussian(variance_type, fixed_variance, left_side, right_side, fixed)
        else:
            return dist2class[structure](left_side, right_side, fixed)

    elif type(structure) == tuple and structure[0] == '+':
        child_models = [get_model_helper(s, left_side, right_side, False, fixed_variance, variance_type)
                        for s in structure[1:]]
        return Sum(child_models, left_side, right_side, fixed)
        
    elif type(structure) == tuple and structure[0] == '*':
        assert len(structure) == 3
        
        iv = continuous_right(structure[1]) and continuous_left(structure[2])
        if iv:
            left_variance_type = 'col'
            right_variance_type = 'row'
        else:
            left_variance_type = right_variance_type = 'scalar'
            
        left_fixed = (structure[2] == 'C')
        right_fixed = (structure[1] == 'c')
        
        left = get_model_helper(structure[1], left_side, False, left_fixed, left_fixed, left_variance_type)
        right = get_model_helper(structure[2], False, right_side, right_fixed, right_fixed, right_variance_type)
        return Product(left, right, left_side, right_side, fixed)

    elif type(structure) == tuple and structure[0] == 's':
        assert len(structure) == 2

        scale_node = get_model_helper(structure[1], left_side, right_side, False, False, 'scalar')
        return GSM(left_side, right_side, fixed, scale_node, variance_type)
    
    else:
        raise RuntimeError('Invalid structure: %s' % structure)
            

def assign_ids(model_node, next_id=1):
    model_node.id = next_id
    next_id += 1
    for child in model_node.children:
        next_id = assign_ids(child, next_id)
    return next_id


def get_model(structure, fixed_noise_variance=False):
    model = get_model_helper(structure, True, True, False, fixed_noise_variance, 'scalar')
    assign_ids(model)
    return model

def align(node, model_node):
    assert node.model is None
    node.model = model_node
    for nchild, mchild in zip(node.children, model_node.children):
        align(nchild, mchild)
    
