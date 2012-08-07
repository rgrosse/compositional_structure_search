import numpy as np
nax = np.newaxis

import grammar
import observations
import recursive
import scoring
import synthetic_experiments

def run_one():
    X = synthetic_experiments.generate_data('sparse', 50, 50, 5)
    dm = observations.DataMatrix.from_real_values(X)
    seq = [grammar.parse('gg+g'), grammar.parse('s(g)g+g')]
    fits = recursive.fit_sequence(seq, dm)
    root = fits[-1]
    scoring.evaluate_model(dm, root, dm, dm)
