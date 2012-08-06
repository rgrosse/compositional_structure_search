import cPickle
import numpy as np
nax = np.newaxis
import os
import pylab
import random
Random = random.Random()
import scipy.linalg

import data
import experiments
import grammar
import mocap
#import models
from utils import misc, storage

def get_sorted_clusters(U):
    nrows, nlat = U.shape
    cluster_ids = [i for i in range(nlat) if np.sum(U[:,i]) > 0]
    cluster_ids.sort(key=lambda i: np.sum(U[:,i]), reverse=True)
    return np.concatenate([np.where(U[:,i])[0] for i in cluster_ids])

def sort_by_largest(factors):
    U, V = factors.children[0].value(), factors.children[1].value()
    k = U.shape[1]
    variance = [np.var(U[:,i]) * np.var(V[i,:]) for i in range(k)]
    comp = np.argmax(variance)
    return np.argsort(U[:,comp])

def dominant_component_indices(factors):
    U, V = factors.children[0].value(), factors.children[1].value()
    k = U.shape[1]
    variance = [np.var(U[:,i]) * np.var(V[i,:]) for i in range(k)]
    return np.argsort(variance)[::-1]

def dominant_component(factors):
    U, V = factors.children[0].value(), factors.children[1].value()
    k = U.shape[1]
    variance = [np.var(U[:,i]) * np.var(V[i,:]) for i in range(k)]
    comp = np.argmax(variance)
    return U[:,comp], V[comp,:]

def sort_by_assignments_and_continuous(u, assignments=None):
    if assignments is None:
        assignments = np.zeros(u.shape)
    cluster_ids = sorted(set(assignments), key=lambda i: np.sum(assignments==i), reverse=True)
    all_rows = []
    for cid in cluster_ids:
        rows = list(np.where(assignments==i))
        rows.sort(lambda i: u[i])
        all_rows += rows
    return all_rows

def show_irm(X, u, v, row_assignments=None, col_assignments=None, spc=1, row_subset=None, col_subset=None,
             sort_clusters_by_eig=False):
    if row_assignments is None:
        row_assignments = np.zeros(u.shape, dtype=int)
    if col_assignments is None:
        col_assignments = np.zeros(v.shape, dtype=int)
    if row_subset is None:
        row_subset = range(u.size)
    if col_subset is None:
        col_subset = range(v.size)

    if sort_clusters_by_eig:
        u, v = np.array(u), np.array(v)
        row_cluster_ids = sorted(set(row_assignments), key=lambda i: np.mean(u[row_assignments==i]))
        col_cluster_ids = sorted(set(col_assignments), key=lambda i: np.mean(v[col_assignments==i]))
    else:
        row_cluster_ids = sorted(set(row_assignments), key=lambda i: np.sum(row_assignments==i), reverse=True)
        col_cluster_ids = sorted(set(col_assignments), key=lambda i: np.sum(col_assignments==i), reverse=True)
    
    row_blocks = []
    for rc in row_cluster_ids:
        rows = filter(lambda i: i in row_subset, np.where(row_assignments==rc)[0])
        rows = np.array(sorted(rows, key=lambda i: u[i]))

        
        col_blocks = []
        for cc in col_cluster_ids:
            cols = filter(lambda i: i in col_subset, np.where(col_assignments==cc)[0])
            cols = np.array(sorted(cols, key=lambda i: v[i]))

            curr_block = np.tile(X[rows[:,nax,nax], cols[nax,:,nax]], [1, 1, 3])
            col_blocks.append(curr_block)
        row_block = vis.hjoin(col_blocks, backcolor=[0,0,1], spc=spc)
        row_blocks.append(row_block)
    vis.display(vis.add_boundary(vis.vjoin(row_blocks, backcolor=[0,0,1], spc=spc), spc=2))
    return vis.add_boundary(vis.vjoin(row_blocks, backcolor=[0,0,1], spc=spc), spc=2)


def visualize_senate_data():
    raise NotImplementedError()
    # as written for NIPS submission, 6-3-10
    data_matrix = data.load_senate_data(2008)
    _, splits = cPickle.load(open(experiments.data_file('senate-2008', 'senate-2008')))
    row_ids, col_ids, _, _ = splits[0]
    #data_matrix = data_matrix.submatrix(row_ids, col_ids)
    data_matrix = data_matrix[row_ids[:, nax], col_ids[nax, :]]
    nrows, ncols = data_matrix.shape
    np.random.seed(0); temp_cols = list(np.random.permutation(ncols))[:200]; np.random.seed()
    #data_matrix = data_matrix.submatrix(range(nrows), temp_cols)

    X = data_matrix.root.value()
    U, d, V = scipy.linalg.svd(X)
    u = U[:,0]
    v = V[0,:]


    # Level 1
    _, _, model = cPickle.load(open(experiments.samples_file('senate-2008', 'senate-2008', grammar.parse('gg+g'), 0, 0)))
    #model = model.submatrix(range(nrows), temp_cols)
    #rows = sort_by_largest(model.root.children[0])
    #cols = sort_by_largest(model.root.children[0].transpose())
    #vis.pw.figure('Level 1: $GG+G$')
    #vis.display(vis.norm01(data_matrix.root.value()[rows[:,nax], cols[nax,:]]))
    #pylab.title('')

    u, v = dominant_component(model.root.children[0])

    vis.figure('Level 1')
    show_irm(-X, u, v, col_subset=temp_cols)
    

    
    # Level 2
    _, _, model = cPickle.load(open(experiments.samples_file('senate-2008-round2b', 'senate-2008',
                                                             grammar.parse('(mg+g)g+g'), 0, 0)))
    #model = model.submatrix(range(nrows), temp_cols)
    assignments = model.root.children[0].children[0].children[0].children[0].value().argmax(1)
    #rows = np.argsort(assignments)
    #cols = sort_by_largest(model.root.children[0].transpose())
    #vis.pw.figure('Level 2: $(MG+G)G+G$')
    #vis.display(vis.norm01(data_matrix.root.value()[rows[:,nax], cols[nax,:]]))
    #pylab.title('')

    u, v = dominant_component(model.root.children[0])

    vis.figure('Level 2')
    show_irm(-X, u, v, assignments, col_subset=temp_cols)


    # Level 2
    _, _, model = cPickle.load(open(experiments.samples_file('senate-2008-round3b', 'senate-2008',
                                                             grammar.parse('(mg+g)(gM+g)+g'), 0, 0)))
    #model = model.submatrix(range(nrows), temp_cols)
    row_assignments = model.root.children[0].children[0].children[0].children[0].value().argmax(1)
    #rows = np.argsort(assignments)

    #rows = get_sorted_clusters(model.root.children[0].children[0].children[0].children[0].value())
    
    col_assignments = model.root.children[0].children[1].children[0].children[1].value().argmax(0)
    #cols = np.argsort(assignments)

    #cols = get_sorted_clusters(model.root.children[0].children[1].children[0].children[1].value().T)

    u, v = dominant_component(model.root.children[0])

    vis.figure('Level 3')
    show_irm(-X, u, v, row_assignments, col_assignments, col_subset=temp_cols)

    #vis.pw.figure('Level 3: $(MG+G)(GM^T+G)+G$')
    #vis.display(vis.norm01(data_matrix.root.value()[rows[:,nax], cols[nax,:]]))

    pylab.title('')

def visualize_senate_data_new():
    raise NotImplementedError()
    # as written for AISTATS submission, 11-1-10
    data_matrix = data.load_senate_data(2008)
    _, splits = cPickle.load(open(experiments.data_file('senate-2008', 'senate-2008')))
    row_ids, col_ids, _, _ = splits[0]
    #data_matrix = data_matrix.submatrix(row_ids, col_ids)
    data_matrix = data_matrix[row_ids[:, nax], col_ids[nax, :]]
    nrows, ncols = data_matrix.shape
    np.random.seed(0); temp_cols = list(np.random.permutation(ncols))[:200]; np.random.seed()
    #data_matrix = data_matrix.submatrix(range(nrows), temp_cols)

    X = data_matrix.root.value()
    U, d, V = scipy.linalg.svd(X)
    u = U[:,0]
    v = V[0,:]


    # Level 1
    _, _, model = cPickle.load(open(experiments.samples_file('senate-2008', 'senate-2008', grammar.parse('gg+g'), 0, 0)))
    #model = model.submatrix(range(nrows), temp_cols)
    #rows = sort_by_largest(model.root.children[0])
    #cols = sort_by_largest(model.root.children[0].transpose())
    #vis.pw.figure('Level 1: $GG+G$')
    #vis.display(vis.norm01(data_matrix.root.value()[rows[:,nax], cols[nax,:]]))
    #pylab.title('')

    u, v = dominant_component(model.root.children[0])

    vis.figure('Level 1')
    img = show_irm(-X, u, v, col_subset=temp_cols, sort_clusters_by_eig=True)
    misc.arr2img(expand(img, 5)).save('/tmp/roger/senate/level1.png')

    
    # Level 2
    _, _, model = cPickle.load(open(experiments.samples_file('senate-2008-round2b', 'senate-2008',
                                                             grammar.parse('(mg+g)g+g'), 0, 0)))
    #model = model.submatrix(range(nrows), temp_cols)
    assignments = model.root.children[0].children[0].children[0].children[0].value().argmax(1)
    #rows = np.argsort(assignments)
    #cols = sort_by_largest(model.root.children[0].transpose())
    #vis.pw.figure('Level 2: $(MG+G)G+G$')
    #vis.display(vis.norm01(data_matrix.root.value()[rows[:,nax], cols[nax,:]]))
    #pylab.title('')

    u, v = dominant_component(model.root.children[0])

    vis.figure('Level 2')
    img = show_irm(-X, -u, -v, assignments, col_subset=temp_cols, sort_clusters_by_eig=True)
    misc.arr2img(expand(img, 5)).save('/tmp/roger/senate/level2.png')


    # Level 2
    _, _, model = cPickle.load(open(experiments.samples_file('senate-2008-round3b', 'senate-2008',
                                                             grammar.parse('(mg+g)(gM+g)+g'), 0, 0)))
    #model = model.submatrix(range(nrows), temp_cols)
    row_assignments = model.root.children[0].children[0].children[0].children[0].value().argmax(1)
    #rows = np.argsort(assignments)

    #rows = get_sorted_clusters(model.root.children[0].children[0].children[0].children[0].value())
    
    col_assignments = model.root.children[0].children[1].children[0].children[1].value().argmax(0)
    #cols = np.argsort(assignments)

    #cols = get_sorted_clusters(model.root.children[0].children[1].children[0].children[1].value().T)

    u, v = dominant_component(model.root.children[0])

    vis.figure('Level 3')
    img = show_irm(-X, u, v, row_assignments, col_assignments, col_subset=temp_cols, sort_clusters_by_eig=True)
    misc.arr2img(expand(img, 5)).save('/tmp/roger/senate/level3.png')

    #vis.pw.figure('Level 3: $(MG+G)(GM^T+G)+G$')
    #vis.display(vis.norm01(data_matrix.root.value()[rows[:,nax], cols[nax,:]]))

    pylab.title('')
    
def visualize_intel_data():
    raise NotImplementedError()
    data_matrix = data.load_intel_data()
    _, splits = cPickle.load(open(experiments.data_file('intel', 'intel')))
    row_ids, col_ids, _, _ = splits[0]
    #data_matrix = data_matrix.submatrix(row_ids, col_ids)
    data_matrix = data_matrix[row_ids[:, nax], col_ids[nax, :]]
    nrows, ncols = data_matrix.shape
    np.random.seed(0); temp_rows = list(np.random.permutation(nrows))[:250]; np.random.seed()
    #data_matrix = data_matrix.submatrix(temp_rows, range(ncols))

    X = data_matrix.root.value()



    # Level 1
    _, _, model = cPickle.load(open(experiments.samples_file('intel', 'intel', grammar.parse('gg+g'), 0, 0)))
    #model = model.submatrix(temp_rows, range(ncols))
    #rows = sort_by_largest(model.root.children[0])
    #cols = sort_by_largest(model.root.children[0].transpose())
    #vis.pw.figure('Level 1: $GG+G$')
    #vis.display(vis.norm01(data_matrix.root.value()[rows[:,nax], cols[nax,:]]))
    #pylab.title('')

    u, v = dominant_component(model.root.children[0])
    vis.figure('Level 1')
    show_irm(-X, u, v, row_subset=temp_rows, spc=2)
    
    # Level 2
    _, _, model = cPickle.load(open(experiments.samples_file('intel-round2', 'intel',
                                                             grammar.parse('(mg+g)g+g'), 0, 0)))
    #model = model.submatrix(temp_rows, range(ncols))
    #assignments = model.root.children[0].children[0].children[0].children[0].value().argmax(1)
    #rows = np.argsort(assignments)
    #rows = get_sorted_clusters(model.root.children[0].children[0].children[0].children[0].value())
    #cols = sort_by_largest(model.root.children[0].transpose())
    #vis.pw.figure('Level 2: $(MG+G)G+G$')
    #vis.display(vis.norm01(data_matrix.root.value()[rows[:,nax], cols[nax,:]]))
    #pylab.title('')

    assignments = model.root.children[0].children[0].children[0].children[0].value().argmax(1)
    u, v = dominant_component(model.root.children[0])
    vis.figure('Level 2')
    show_irm(-X, u, v, assignments, row_subset=temp_rows, spc=2)


def visualize_intel_data_new():
    raise NotImplementedError()
    data_matrix = data.load_intel_data()
    _, splits = cPickle.load(open(experiments.data_file('intel', 'intel')))
    row_ids, col_ids, _, _ = splits[0]
    #data_matrix = data_matrix.submatrix(row_ids, col_ids)
    data_matrix = data_matrix[row_ids[:, nax], col_ids[nax, :]]
    nrows, ncols = data_matrix.shape
    np.random.seed(0); temp_rows = list(np.random.permutation(nrows))[:250]; np.random.seed()
    #data_matrix = data_matrix.submatrix(temp_rows, range(ncols))

    X = data_matrix.root.value()



    # Level 1
    _, _, model = cPickle.load(open(experiments.samples_file('intel', 'intel', grammar.parse('gg+g'), 0, 0)))
    #model = model.submatrix(temp_rows, range(ncols))
    #rows = sort_by_largest(model.root.children[0])
    #cols = sort_by_largest(model.root.children[0].transpose())
    #vis.pw.figure('Level 1: $GG+G$')
    #vis.display(vis.norm01(data_matrix.root.value()[rows[:,nax], cols[nax,:]]))
    #pylab.title('')

    u, v = dominant_component(model.root.children[0])
    vis.figure('Level 1')
    img = show_irm(-X, u, v, row_subset=temp_rows, spc=2)
    misc.arr2img(img).save('/tmp/roger/intel/level1.png')
    
    # Level 2
    _, _, model = cPickle.load(open(experiments.samples_file('intel-round2', 'intel',
                                                             grammar.parse('(mg+g)g+g'), 0, 0)))
    #model = model.submatrix(temp_rows, range(ncols))
    #assignments = model.root.children[0].children[0].children[0].children[0].value().argmax(1)
    #rows = np.argsort(assignments)
    #rows = get_sorted_clusters(model.root.children[0].children[0].children[0].children[0].value())
    #cols = sort_by_largest(model.root.children[0].transpose())
    #vis.pw.figure('Level 2: $(MG+G)G+G$')
    #vis.display(vis.norm01(data_matrix.root.value()[rows[:,nax], cols[nax,:]]))
    #pylab.title('')

    assignments = model.root.children[0].children[0].children[0].children[0].value().argmax(1)
    u, v = dominant_component(model.root.children[0])
    vis.figure('Level 2')
    img = show_irm(-X, u, v, assignments, row_subset=temp_rows, spc=2, sort_clusters_by_eig=True)
    misc.arr2img(img).save('/tmp/roger/intel/level2.png')




def print_most_central():
    raise NotImplementedError()
    data_matrix = data.load_intel_data()
    _, splits = cPickle.load(open(experiments.data_file('intel', 'intel')))
    row_ids, col_ids, _, _ = splits[0]
    #data_matrix = data_matrix.submatrix(row_ids, col_ids)
    data_matrix = data_matrix[row_ids[:, nax], col_ids[nax, :]]
    nrows, ncols = data_matrix.shape
    X = data_matrix.root.value()
    objects = data.load_intel_objects()
    objects = [objects[r] for r in row_ids]

    _, _, model = cPickle.load(open(experiments.samples_file('intel-round2', 'intel',
                                                             grammar.parse('(mg+g)g+g'), 0, 0)))
    assignments = model.root.children[0].children[0].children[0].children[0].value().argmax(1)
    cluster_ids = sorted(set(assignments), key=lambda i: np.sum(assignments==i), reverse=True)
    
    temp1 = model.root.children[0].children[0].children[0].children[1].value()
    temp2 = model.root.children[0].children[1].value()
    centers = np.dot(temp1, temp2)
    for c in cluster_ids:
        dist = np.sum((model.root.value() - centers[c,:][nax,:])**2, axis=1)
        #dist = np.sum(model.root.children[0].children[0].children[1].value()**2, axis=1)
        rows = np.where(assignments==c)[0]
        rows = sorted(rows, key=lambda i: dist[i])
        for r in rows:
            print objects[r]

                      
        print
        print

    assert False
    
        
def mocap_figure():
    raise NotImplementedError()
    _, splits = cPickle.load(open(experiments.data_file('mocap', 'mocap')))
    row_ids = np.array(splits[0][0])
    time_steps = row_ids[1:] - row_ids[:-1]
    _, _, model = cPickle.load(open(experiments.samples_file('mocap-round2', 'mocap',
                                                             grammar.parse('c(gg+g)+g'), 0, 0)))
    temp = dominant_component_indices(model.root.children[0].children[1].children[0])
    k1, k2 = temp[0], temp[1]

    derivs1 = model.root.children[0].children[1].children[0].children[0].value()[:, k1]
    derivs2 = model.root.children[0].children[1].children[0].children[0].value()[:, k2]
    total1 = derivs1.cumsum(0)
    total2 = derivs2.cumsum(0)

    vis.figure('Derivatives')
    pylab.clf()
    pylab.title('Derivatives')
    h1 = pylab.plot(row_ids, derivs2, 'r-')
    derivs2[1:] /= time_steps
    h2 = pylab.plot(row_ids, derivs2, 'b-')
    pylab.legend([h1, h2], ['before', 'after'])
    
    vis.figure('Total')
    pylab.clf()
    pylab.title('Total')
    h1 = pylab.plot(row_ids, total1, 'r-')


def mocap_figure2():
    raise NotImplementedError()
    _, splits = cPickle.load(open(experiments.data_file('mocap', 'mocap')))
    row_ids = np.array(splits[0][0])
    time_steps = row_ids[1:] - row_ids[:-1]
    _, _, model = cPickle.load(open(experiments.samples_file('mocap-round2', 'mocap',
                                                             grammar.parse('c(gg+g)+g'), 0, 0)))

    #transition_matrix = model.root.children[0].children[1]
    #transition_matrix = model.root.children[0].children[1].children[0]
    transition_matrix = model.root.children[0]
    T = transition_matrix.value()[1:,:]
    #T /= time_steps[:,nax]
    U, d, Vt = scipy.linalg.svd(T)

    print 'd', d
    vis.figure('Principal components of transition matrix')
    pylab.clf()
    pylab.plot(U[:,0], U[:,1])


    
    


def show_binary_matrix(B, block_size, spc):
    m, n = B.shape
    result = 0.25 * np.ones((block_size*m + spc*(m+1), block_size*n + spc*(n+1)))
    for i in range(m):
        for j in range(n):
            top = (i+1)*spc + i*block_size
            bottom = (i+1)*spc + (i+1)*block_size
            left = (j+1)*spc + (j)*block_size
            right = (j+1)*spc + (j+1)*block_size
            result[top:bottom, left:right] = B[i,j]
    return result

def expand(B, mult=25):
    m, n = B.shape[:2]
    result = np.zeros((mult*m, mult*n) + B.shape[2:])
    for i in range(m):
        for j in range(n):
            result[i*mult:(i+1)*mult, j*mult:(j+1)*mult,...] = B[i,j,...]
    return result

def irm_matrices():
    row_assignments = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    left_M = np.zeros((row_assignments.size, np.max(row_assignments)+1))
    left_M[np.arange(row_assignments.size), row_assignments] = 1.
    col_assignments = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3])
    right_M = np.zeros((np.max(col_assignments) + 1, col_assignments.size))
    right_M[col_assignments, np.arange(col_assignments.size)] = 1.

    vis.pw.figure('left_M')
    vis.display(vis.norm01(show_binary_matrix(1-left_M, 10, 1)))
    misc.arr2img(vis.norm01(show_binary_matrix(1-left_M, 10, 1))).save('/tmp/roger/toy-matrices/left-M.png')
    
    vis.pw.figure('right_M')
    vis.display(vis.norm01(show_binary_matrix(1-right_M, 10, 1)))
    misc.arr2img(vis.norm01(show_binary_matrix(1-right_M, 10, 1))).save('/tmp/roger/toy-matrices/right-M.png')

    centers = np.random.normal(size=(np.max(row_assignments)+1, np.max(col_assignments)+1))
    vis.pw.figure('centers')
    vis.display(vis.norm01(centers))
    misc.arr2img(expand(vis.norm01(centers))).save('/tmp/roger/toy-matrices/centers.png')

    R = np.random.normal(size=(row_assignments.size, col_assignments.size), scale=0.2)
    R_ = np.tile(R[:,:,nax], [1, 1, 3])
    vis.pw.figure('R')
    vis.display(0.1 * vis.norm01(R_) + 0.5)
    misc.arr2img(expand(0.1 * vis.norm01(R_) + 0.5)).save('/tmp/roger/toy-matrices/R.png')

    final = centers[row_assignments[:,nax], col_assignments[nax,:]] + R
    vis.pw.figure('final')
    vis.display(vis.norm01(final))
    misc.arr2img(expand(vis.norm01(final))).save('/tmp/roger/toy-matrices/final.png')
    

def cartoon_matrices():
    # gaussian
    misc.arr2img(expand(vis.norm01(np.random.normal(size=(12, 6))))).save('/tmp/roger/cartoon-matrices/G.png')

    # multinomial
    pi = distributions_old.dirichlet(6, 1.)
    mat = np.random.multinomial(1, pi, size=12)
    misc.arr2img(vis.norm01(show_binary_matrix(1-mat, 10, 1))).save('/tmp/roger/cartoon-matrices/M.png')

    # binary
    pi = np.random.beta(1., 2., size=6)
    mat = np.random.binomial(1, pi[nax,:], size=(12, 6))
    misc.arr2img(vis.norm01(show_binary_matrix(1-mat, 10, 1))).save('/tmp/roger/cartoon-matrices/B.png')

    # integration
    C = (np.arange(12)[:,nax] >= np.arange(12)[nax,:])
    misc.arr2img(vis.norm01(show_binary_matrix(1-C, 10, 1))).save('/tmp/roger/cartoon-matrices/C.png')

def cartoon_binary(path):
    N, K, D = 20, 8, 25
    
    # binary
    pi = np.random.beta(1., 1.5, size=K)
    mat = np.random.binomial(1, pi[nax,:], size=(N, K))
    fname = os.path.join(path, 'B.png')
    misc.arr2img(vis.norm01(show_binary_matrix(1-mat, 10, 1))).save(fname)

    # gaussian
    fname = os.path.join(path, 'G1.png')
    misc.arr2img(expand(vis.norm01(np.random.normal(size=(K, D))))).save(fname)

    # gaussian
    fname = os.path.join(path, 'G2.png')
    misc.arr2img(expand(0.3 + 0.4 * vis.norm01(np.random.normal(size=(N, D))))).save(fname)

    



def cartoon_data():
    data_matrix = data.load_animals_data()
    X = vis.norm01(expand(-data_matrix.root.value(), 5))
    misc.arr2img(X).save('/tmp/roger/cartoon-animals.png')
    vis.figure('animals')
    vis.display(X)

    data_matrix = mocap.load_data()
    X = vis.norm01(expand(data_matrix[:450:3, :], 5))
    misc.arr2img(X).save('/tmp/roger/cartoon-mocap.png')
    vis.figure('mocap')
    vis.display(X)

    data_matrix = data.load_senate_data(2008)
    X = vis.norm01(expand(-data_matrix.root.value()[:,:160], 5))
    misc.arr2img(X).save('/tmp/roger/cartoon-senate.png')
    vis.figure('senate')
    vis.display(X)


def no_scaling(arr):
    return np.tile(arr[:,:,nax], [1, 1, 3])

def irm_matrices2():
    row_assignments = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    col_assignments = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3])

    num_row_clusters = np.max(row_assignments) + 1
    num_col_clusters = np.max(col_assignments) + 1
    num_rows = row_assignments.size
    num_cols = col_assignments.size
    
    left_M = np.zeros((num_rows, num_row_clusters))
    left_M[np.arange(num_rows), row_assignments] = 1.

    right_M = np.zeros((num_col_clusters, num_cols))
    right_M[col_assignments, np.arange(num_cols)] = 1.

    

    vis.pw.figure('left_M')
    vis.display(vis.norm01(show_binary_matrix(1-left_M, 10, 1)))
    misc.arr2img(vis.norm01(show_binary_matrix(1-left_M, 10, 1))).save('/tmp/roger/toy-matrices/left-M.png')
    
    vis.pw.figure('right_M')
    vis.display(vis.norm01(show_binary_matrix(1-right_M, 10, 1)))
    misc.arr2img(vis.norm01(show_binary_matrix(1-right_M, 10, 1))).save('/tmp/roger/toy-matrices/right-M.png')

    centers = np.random.uniform(0.2, 0.8, size=(num_row_clusters, num_col_clusters))
    vis.pw.figure('centers')
    vis.display(no_scaling(centers))
    misc.arr2img(expand(centers)).save('/tmp/roger/toy-matrices/centers.png')

    inner_noise = np.random.uniform(-0.025, 0.025, size=(num_row_clusters, num_cols))
    vis.pw.figure('inner noise')
    vis.display(no_scaling(0.5 + inner_noise))
    misc.arr2img(expand(0.5 + inner_noise)).save('/tmp/roger/toy-matrices/inner-noise.png')

    row_centers = np.dot(centers, right_M) + inner_noise
    vis.pw.figure('row centers')
    vis.display(no_scaling(row_centers))
    misc.arr2img(expand(row_centers)).save('/tmp/roger/toy-matrices/row-centers.png')

    outer_noise = np.random.uniform(-0.075, 0.075, size=(num_rows, num_cols))
    vis.pw.figure('outer noise')
    vis.display(no_scaling(0.5 + outer_noise))
    misc.arr2img(expand(0.5 + outer_noise)).save('/tmp/roger/toy-matrices/outer-noise.png')

    #final = centers[row_assignments[:,nax], col_assignments[nax,:]] + R
    final = np.dot(left_M, row_centers) + outer_noise
    vis.pw.figure('final')
    vis.display(no_scaling(final))
    misc.arr2img(expand(final)).save('/tmp/roger/toy-matrices/final.png')

    
def run_noise_levels():
    row_assignments = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    col_assignments = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3])


    row_assignments = np.array([row_assignments, row_assignments]).T.ravel()
    col_assignments = np.array([col_assignments, col_assignments]).T.ravel()

    

    num_row_clusters = np.max(row_assignments) + 1
    num_col_clusters = np.max(col_assignments) + 1
    num_rows = row_assignments.size
    num_cols = col_assignments.size
    
    left_M = np.zeros((num_rows, num_row_clusters))
    left_M[np.arange(num_rows), row_assignments] = 1.

    right_M = np.zeros((num_col_clusters, num_cols))
    right_M[col_assignments, np.arange(num_cols)] = 1.

    centers = np.random.normal(size=(num_row_clusters, num_col_clusters))


    #row_centers = np.dot(centers, right_M) + inner_noise

    #outer_noise = np.random.uniform(-0.075, 0.075, size=(num_rows, num_cols))

    signal = centers[row_assignments[:,nax], col_assignments[nax,:]]
    noise = np.random.normal(size=(num_rows, num_cols))

    for noise_id, sigma_sq in enumerate([0.1, 1., 3., 10.]):
        result = signal + np.sqrt(sigma_sq) * noise
        title = 'Input matrix, sigma^2=%1.1f' % sigma_sq
        vis.pw.figure(title)
        vis.display(vis.norm01(result))
        misc.arr2img(expand(vis.norm01(result))).save('/tmp/roger/toy-matrices/noise-%d.png' % noise_id)
        
    
    
def visualize_gg(data_matrix, root, name, rows, cols):
    X = data_matrix.observations.values.copy()
    u, v = dominant_component(root.children[0])

    vis.figure('Level 1')
    img = show_irm(-X, u, v, row_subset=rows, col_subset=cols, sort_clusters_by_eig=True)
    misc.arr2img(expand(img, 5)).save('/users/rgrosse/desktop/temp-figures/%s-gg.png' % name)

def visualize_mg(data_matrix, root, name, rows, cols):
    X = data_matrix.observations.values.copy()
    u, v = dominant_component(root.children[0])

    row_assignments = root.children[0].children[0].children[0].children[0].value().argmax(1)

    vis.figure('Level 2')
    img = show_irm(-X, -u, -v, row_assignments, row_subset=rows, col_subset=cols, sort_clusters_by_eig=True)
    misc.arr2img(expand(img, 5)).save('/users/rgrosse/desktop/temp-figures/%s-mg.png' % name)

def visualize_mm(data_matrix, root, name, rows, cols):
    X = data_matrix.observations.values.copy()
    u, v = dominant_component(root.children[0])

    row_assignments = root.children[0].children[0].children[0].children[0].value().argmax(1)
    col_assignments = root.children[0].children[1].children[0].children[1].value().argmax(0)

    vis.figure('Level 3')
    img = show_irm(-X, -u, -v, row_assignments, col_assignments, row_subset=rows, col_subset=cols, sort_clusters_by_eig=True)
    misc.arr2img(expand(img, 5)).save('/users/rgrosse/desktop/temp-figures/%s-mm.png' % name)
    

def visualize_senate(seq):
    data_matrix = data.load_senate_data(2008)
    cols = np.random.permutation(data_matrix.n)[:200]
    visualize_gg(data_matrix, seq[1], 'senate', range(data_matrix.m), cols)
    visualize_mg(data_matrix, seq[2], 'senate', range(data_matrix.m), cols)
    visualize_mm(data_matrix, seq[3], 'senate', range(data_matrix.m), cols)

def visualize_senate2():
    name = 'senate'
    split_id = 0
    sample_id = 0
    
    data_matrix = experiments.load_data(name)
    splits = storage.load(experiments.splits_file(name))
    train_rows, train_cols, test_rows, test_cols = splits[split_id]
    
    X_train = data_matrix[train_rows[:, nax], train_cols[nax, :]]

    level = 1
    structure = grammar.parse('gg+g')
    
    sample1 = storage.load(experiments.samples_file(name, level, structure, split_id, sample_id))

    visualize_gg(X_train, sample1, 'senate', range(data_matrix.m), cols)
    visualize_mg(X_train, sample2, 'senate', range(data_matrix.m), cols)
    visualize_mm(X_train, sample3, 'senate', range(data_matrix.m), cols)

def visualize_senate3():
    data_matrix = data.load_senate_data(2008)
    root = cPickle.load(open('/users/rgrosse/desktop/senate-root.pickle'))
    
    cols = np.random.permutation(data_matrix.n)[:200]
    visualize_gg(data_matrix, root, 'senate', range(data_matrix.m), cols)
    visualize_mg(data_matrix, root, 'senate', range(data_matrix.m), cols)
    visualize_mm(data_matrix, root, 'senate', range(data_matrix.m), cols)


    
    
def visualize_intel(seq):
    data_matrix = data.load_intel_data()
    rows = np.random.permutation(data_matrix.m)[:250]
    cols = np.random.permutation(data_matrix.n)[:100]
    visualize_gg(data_matrix, seq[1], 'intel', rows, cols)
    visualize_mg(data_matrix, seq[2], 'intel', rows, cols)
    
