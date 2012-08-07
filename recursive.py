import numpy as np
nax = np.newaxis
import utils

from algorithms import chains, dumb_samplers
import initialization
import models





######################### Sub-expressions ######################################

class Decomp:
    pass       # keep this around so that old pickled data can be loaded


class Node:
    def root(self):
        if self.parent is not None:
            return self.parent.root()
        else:
            return self

    def isroot(self):
        return self.parent is None

    def isproduct(self):
        return False
    def issum(self):
        return False
    def isleaf(self):
        return False
    def isgsm(self):
        return False

    def descendant(self, path):
        return descendant(self, path)


class LeafNode(Node):
    def __init__(self, value):
        self.set_value(value)
        self.m, self.n = value.shape
        self.children = []
        self.parent = None
        self.model = None

    def description(self):
        return 'Leaf(%s)' % self.distribution()

    def value(self):
        return self._value.copy()

    def set_value(self, value):
        self._value = value.copy()

    def copy(self):
        return self.__class__(self.value())

    def __getitem__(self, slc):
        return self.__class__(self.value()[slc].copy())

    def fits_assumptions(self):
        return True

    def transpose(self):
        return self.transpose_class()(self.value().T)

    def structure(self):
        return self.distribution()

    

    def has_children(self):
        return False

    def gibbs_update(self, U, V, X):
        return U

    def gibbs_update2(self):
        pass

    def isleaf(self):
        return True


        

class GaussianNode(LeafNode):
    def __init__(self, value, variance_type, sigma_sq):
        LeafNode.__init__(self, value)
        if variance_type not in ['scalar', 'row', 'col']:
            raise RuntimeError('Unknown variance type: %s' % variance_type)
        self.variance_type = variance_type
        self.sigma_sq = sigma_sq
        
    def distribution(self):
        return 'g'

    def has_rank1_variance(self):
        return True

    def variance(self):
        sigma_sq_row, sigma_sq_col = self.row_col_variance()
        return np.outer(sigma_sq_row, sigma_sq_col)

    def row_col_variance(self):
        if self.variance_type == 'row':
            sigma_sq_row = self.sigma_sq
            sigma_sq_col = np.ones(self.n)
        elif self.variance_type == 'col':
            sigma_sq_row = np.ones(self.m)
            sigma_sq_col = self.sigma_sq
        elif self.variance_type == 'scalar':
            sigma_sq_row = self.sigma_sq * np.ones(self.m)
            sigma_sq_col = np.ones(self.n)
        return sigma_sq_row.copy(), sigma_sq_col.copy()

    def sample_variance(self):
        if self.variance_type == 'scalar':
            a = 0.01 + 0.5 * self.m * self.n
            b = 0.01 + 0.5 * np.sum(self.value() ** 2)
        elif self.variance_type == 'row':
            a = 0.01 + 0.5 * self.n
            b = 0.01 + 0.5 * np.sum(self.value() ** 2, axis=1)
        elif self.variance_type == 'col':
            a = 0.01 + 0.5 * self.m
            b = 0.01 + 0.5 * np.sum(self.value() ** 2, axis=0)
        self.sigma_sq = 1. / np.random.gamma(a, 1. / b)
        

    def transpose_class(self):
        return GaussianNode

    def copy(self):
        if self.variance_type == 'scalar':
            sigma_sq = self.sigma_sq
        else:
            sigma_sq = self.sigma_sq.copy()
        return GaussianNode(self._value.copy(), self.variance_type, sigma_sq)

    def __getitem__(self, slc):
        rslc, cslc = utils.extract_slices(slc)
        if self.variance_type == 'scalar':
            sigma_sq = self.sigma_sq
        elif self.variance_type == 'row':
            sigma_sq = self.sigma_sq[rslc].copy()
        elif self.variance_type == 'col':
            sigma_sq = self.sigma_sq[cslc].copy()
        return GaussianNode(self._value[slc].copy(), self.variance_type, sigma_sq)

    def transpose(self):
        if self.variance_type == 'scalar':
            variance_type = 'scalar'
            sigma_sq = self.sigma_sq
        elif self.variance_type == 'row':
            variance_type = 'col'
            sigma_sq = self.sigma_sq.copy()
        elif self.variance_type == 'col':
            variance_type = 'row'
            sigma_sq = self.sigma_sq.copy()
        return GaussianNode(self._value.T.copy(), variance_type, sigma_sq)

    @staticmethod
    def dummy(variance_type):
        if variance_type == 'scalar':
            var = 1.
        else:
            var = np.ones(5)
        return GaussianNode(np.zeros((5, 5)), 'scalar', var)


class MultinomialNode(LeafNode):
    def distribution(self):
        return 'm'

    def transpose_class(self):
        return MultinomialTNode

    @staticmethod
    def dummy():
        return MultinomialNode(np.eye(5, dtype=int))


class MultinomialTNode(LeafNode):
    def distribution(self):
        return 'M'

    def transpose_class(self):
        return MultinomialNode

    @staticmethod
    def dummy():
        return MultinomialTNode(np.eye(5, dtype=int))


class BernoulliNode(LeafNode):
    def distribution(self):
        return 'b'

    def transpose_class(self):
        return BernoulliTNode

    @staticmethod
    def dummy():
        return BernoulliNode(np.zeros((5, 5), dtype=int))

class BernoulliTNode(LeafNode):
    def distribution(self):
        return 'B'

    def transpose_class(self):
        return BernoulliNode

    @staticmethod
    def dummy():
        return BernoulliTNode(np.zeros((5, 5), dtype=int))

class IntegrationNode(LeafNode):
    def distribution(self):
        return 'c'

    def transpose_class(self):
        return IntegrationTNode

    @staticmethod
    def dummy():
        return IntegrationNode(chains.integration_matrix(5))

class IntegrationTNode(LeafNode):
    def distribution(self):
        return 'C'

    def transpose_class(self):
        return IntegrationNode

    @staticmethod
    def dummy():
        return IntegrationTNode(chains.integration_matrix(5).T)


class GSMNode(Node):
    def __init__(self, _value, scale_node, bias_type, bias):
        self._value = _value
        self.m, self.n = self._value.shape
        self.parent = None
        self.model = None
        self.scale_node = scale_node
        if bias_type not in ['row', 'col', 'scalar']:
            raise RuntimeError('Unknown bias type: %s' % bias_type)
        self.bias_type = bias_type
        self.bias = bias
        if self.bias_type == 'row':
            self.bias = self.bias.reshape((-1, 1))
        elif self.bias_type == 'col':
            self.bias = self.bias.reshape((1, -1))
        self.children = [self.scale_node]
        self.scale_node.parent = self

    def has_rank1_variance(self):
        return False

    def variance(self):
        return np.exp(self.bias + self.scale_node.value())

    def value(self):
        return self._value.copy()

    def set_value(self, value):
        self._value = value.copy()

    def copy(self):
        if np.isscalar(self.bias):
            return GSMNode(self._value.copy(), self.scale_node.copy(), self.bias_type, self.bias)
        else:
            return GSMNode(self._value.copy(), self.scale_node.copy(), self.bias_type, self.bias.copy())

    def __getitem__(self, slc):
        rslc, cslc = utils.extract_slices(slc)
        if self.bias_type == 'row':
            return GSMNode(self._value[slc].copy(), self.scale_node[slc].copy(), self.bias_type, self.bias[rslc].copy())
        elif self.bias_type == 'col':
            return GSMNode(self._value[slc].copy(), self.scale_node[slc].copy(), self.bias_type, self.bias[cslc].copy())
        elif self.bias_type == 'scalar':
            return GSMNode(self._value[slc].copy(), self.scale_node[slc].copy(), self.bias_type, self.bias)

    def transpose(self):
        if self.bias_type == 'scalar':
            return GSMNode(self._value.T.copy(), self.scale_node.transpose(), 'scalar', self.bias)
        elif self.bias_type == 'row':
            return GSMNode(self._value.T.copy(), self.scale_node.transpose(), 'col', self.bias.T.copy())
        elif self.bias_type == 'col':
            return GSMNode(self._value.T.copy(), self.scale_node.transpose(), 'row', self.bias.T.copy())

    def structure(self):
        return ('s', self.scale_node.structure())

    def has_children(self):
        return True

    def isgsm(self):
        return True


class SumNode(Node):
    def __init__(self, children):
        self.children = children
        self.m = self.children[0].m
        self.n = self.children[0].n
        self.parent = None
        for child in children:
            assert child.parent is None
            child.parent = self
        self.model = None

    def description(self):
        children_str = ', '.join([c.description() for c in self.children])
        return 'Sum(%s)' % children_str


    def value(self):
        return sum([c.value() for c in self.children])

    def copy(self):
        children = [child.copy() for child in self.children]
        other = SumNode(children)
        return other

    def __getitem__(self, slc):
        children = [child[slc] for child in self.children]
        return SumNode(children)


    def fits_assumptions(self):
        for child in self.children[:-1]:
            if not child.isproduct():
                return False
        if not self.children[-1].isleaf():
            return False
        if self.children[-1].distribution() != 'g':
            return False
        return True

    def transpose(self):
        new_model = SumNode([child.transpose() for child in self.children])
        return new_model

    def structure(self):
        return ('+',) + tuple([child.structure() for child in self.children])


    def has_children(self):
        return True

    def issum(self):
        return True

    def predictions(self):
        return self.value() - self.children[-1].value()


class ProductNode(Node):
    def __init__(self, children):
        self.children = children
        self.m = self.children[0].m
        self.n = self.children[-1].n
        self.parent = None
        for child in children:
            assert child.parent is None
            child.parent = self
        self.model = None
        

    def description(self):
        children_str = ', '.join([c.description() for c in self.children])
        return 'Product(%s)' % children_str

    def value(self):
        return utils.mult([child.value() for child in self.children])

    def copy(self):
        children = [child.copy() for child in self.children]
        other = ProductNode(children)
        return other


    def __getitem__(self, slc):
        assert len(self.children) == 2
        rslc, cslc = utils.extract_slices(slc)
        return ProductNode([self.children[0][rslc, :], self.children[1][:, cslc]])


    def fits_assumptions(self):
        if len(self.children) != 2:
            return False
        for child in self.children:
            if not (child.isleaf() or child.issum()):
                return False
        return True

    def transpose(self):
        new_model = ProductNode([child.transpose() for child in self.children[::-1]])
        return new_model

    def structure(self):
        return ('*', self.children[0].structure(), self.children[1].structure())


    def has_children(self):
        return True

    def isproduct(self):
        return True


def get_path(top, bottom):
    if top is bottom:
        return [bottom]
    if top.has_children():
        for child in top.children:
            result = get_path(child, bottom)
            if result:
                return [top] + result
    return None


def compute_row_ids(top, row_ids, bottom):
    path = get_path(top, bottom)
    assert path is not None
    for curr_node, next_node in zip(path[:-1], path[1:]):
        assert curr_node.issum() or curr_node.isproduct() or curr_node.isgsm()
        if curr_node.isproduct():
            assert len(curr_node.children) == 2 and next_node in curr_node.children
            #curr_node.children[0].distribution() != 'c':
            if next_node is curr_node.children[1] and not isinstance(curr_node.children[0], IntegrationNode):
                row_ids = range(curr_node.children[1].m)
    return row_ids

def row_ids_for(data_matrix, node):
    return compute_row_ids(node.root(), data_matrix.row_ids, node)

def compute_col_ids(top, col_ids, bottom):
    path = get_path(top, bottom)
    assert path is not None
    for curr_node, next_node in zip(path[:-1], path[1:]):
        assert curr_node.issum() or curr_node.isproduct() or curr_node.isgsm()
        if curr_node.isproduct():
            assert len(curr_node.children) == 2 and next_node in curr_node.children
            # curr_node.children[1].distribution() != 'C':
            if next_node is curr_node.children[0] and not isinstance(curr_node.children[1], IntegrationTNode):
                col_ids = range(curr_node.children[1].m)
    return col_ids
    
def col_ids_for(data_matrix, node):
    return compute_col_ids(node.root(), data_matrix.col_ids, node)

def compute_orig_shape(top, m_orig, n_orig, bottom):
    path = get_path(top, bottom)
    assert path is not None
    for curr_node, next_node in zip(path[:-1], path[1:]):
        assert curr_node.issum() or curr_node.isproduct() or curr_node.isgsm()
        if curr_node.isproduct():
            assert len(curr_node.children) == 2 and next_node in curr_node.children
            if next_node is curr_node.children[1] and not isinstance(curr_node.children[0], IntegrationNode):
                m_orig = next_node.m
            if next_node is curr_node.children[0] and not isinstance(curr_node.children[1], IntegrationTNode):
                n_orig = next_node.n
    return m_orig, n_orig

def orig_shape_for(data_matrix, node):
    return compute_orig_shape(node.root(), data_matrix.m_orig, data_matrix.n_orig, node)

def find_changed_node(node, old_structure, new_structure):
    assert old_structure != new_structure
    if type(old_structure) == str:
        return node, old_structure, new_structure
    if old_structure[0] != new_structure[0]:
        return node, old_structure, new_structure
    if len(old_structure) != len(new_structure):
        assert old_structure[0] == new_structure[0] == '+'
        assert len(new_structure) == len(old_structure) + 1
        assert old_structure[:-1] == new_structure[:-2]
        return node.children[-1], old_structure[-1], ('+',) + tuple(new_structure[-2:])
    for child, old_str_child, new_str_child in zip(node.children, old_structure[1:], new_structure[1:]):
        if old_str_child != new_str_child:
            return find_changed_node(child, old_str_child, new_str_child)
    assert False

def splice(root, old_node, new_node):
    if old_node is root:
        return new_node
    parent = old_node.parent
    ind = parent.children.index(old_node)
    if parent.issum() and new_node.issum():
        parent.children[ind:ind+1] = new_node.children
        for c in new_node.children:
            c.parent = parent
    elif parent.isgsm():
        parent.scale_node = parent.children[0] = new_node
        new_node.parent = parent
    else:
        parent.children[ind] = new_node
        new_node.parent = parent
    return root

def descendant(node, path):
    if path == '':
        return node
    elif path[0] == 'l':
        return descendant(node.children[0], path[1:])
    elif path[0] == 'r':
        return descendant(node.children[1], path[1:])
    else:
        return descendant(node.children[int(path[0])], path[1:])


# dummy classes for loading old decompositions
LeafModel = LeafNode
SumModel = SumNode
ProductModel = ProductNode



    

   
######################### High-level utilities #################################

def fit_model(structure, data_matrix, old_root=None, gibbs_steps=200):
    if old_root is None:
        X = data_matrix.sample_latent_values(np.zeros((data_matrix.m, data_matrix.n)), 1.)
        old_root = GaussianNode(X, 'scalar', 1.)
    root = initialization.initialize(data_matrix, old_root, old_root.structure(), structure, num_iter=gibbs_steps)
    model = models.get_model(structure, fixed_noise_variance=data_matrix.fixed_variance())
    models.align(root, model)
    dumb_samplers.sweep(data_matrix, root, num_iter=gibbs_steps)
    dumb_samplers.sweep(data_matrix, root, maximize=True, num_iter=1)  
    return root

def fit_sequence(sequence, data_matrix, gibbs_steps=200):
    X = data_matrix.sample_latent_values(np.zeros((data_matrix.m, data_matrix.n)), 1.)
    root = GaussianNode(X, 'scalar', 1.)

    seq = [root.copy()]
    for structure in sequence:
        root = fit_model(structure, data_matrix, root, gibbs_steps)
        seq.append(root.copy())
    return seq





def get_sorted_clusters(U):
    nrows, nlat = U.shape
    cluster_ids = [i for i in range(nlat) if np.sum(U[:,i]) > 0]
    cluster_ids.sort(key=lambda i: np.sum(U[:,i]), reverse=True)
    return [np.where(U[:,i])[0] for i in cluster_ids]

def print_clusters_helper(U, names):
    for c, ids in enumerate(get_sorted_clusters(U)):
        print '  Cluster %d:' % c
        for i in ids:
            print '    %s' % names[i]
        print

def find_nodes(node, property):
    print node
    if property(node):
        found = [node]
    else:
        found = []
    for child in node.children:
        found += find_nodes(child, property)
    return found

def print_clusters(data_matrix, root):
    mb_nodes = find_nodes(root, lambda node: isinstance(node, LeafNode) and node.distribution() in ['m', 'b']
                          and node.m == data_matrix.m)
    print mb_nodes
    for node in mb_nodes:
        row_names = data_matrix.row_labels
        #print location_string(root, node)
        print_clusters_helper(node.value(), row_names)
    MB_nodes = find_nodes(root, lambda node: isinstance(node, LeafNode) and node.distribution() in ['M', 'B']
                          and node.n == data_matrix.n)
    for node in MB_nodes:
        col_names = data_matrix.col_labels
        print_clusters_helper(node.value().T, col_names)


