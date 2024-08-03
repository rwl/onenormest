// Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
// All rights reserved.

// See scipy/sparse/linalg/_onenormest.py in SciPy v1.7.1.

use std::convert::identity;
use std::iter::zip;

use anyhow::{format_err, Result};
use faer::{ColRef, Mat, MatMut, MatRef};
use rand::Rng;

pub const DEFAULT_TRADEOFF: usize = 2;
pub const DEFAULT_IT_MAX: usize = 5;

pub trait Matrix {
    fn n(&self) -> usize;
    fn no_trans(&self, x: MatRef<f64>) -> Mat<f64>;
    fn trans(&self, x: MatRef<f64>) -> Mat<f64>;
}

/// Computes a lower bound of the 1-norm of a sparse matrix.
///
/// Takes a positive parameter `t` controlling the tradeoff between accuracy
/// versus time and memory usage. Larger values take longer and use more
/// memory but give more accurate output. Uses at most `it_max` iterations.
///
/// This is algorithm 2.4 of `[1]`.
///
/// In `[2]` it is described as follows.
///
/// "This algorithm typically requires the evaluation of
/// about 4t matrix-vector products and almost invariably
/// produces a norm estimate (which is, in fact, a lower
/// bound on the norm) correct to within a factor 3."
///
/// ```txt
///  [1] Nicholas J. Higham and Francoise Tisseur (2000),
///      "A Block Algorithm for Matrix 1-Norm Estimation,
///      with an Application to 1-Norm Pseudospectra."
///      SIAM J. Matrix Anal. Appl. Vol. 21, No. 4, pp. 1185-1201.
///
///  [2] Awad H. Al-Mohy and Nicholas J. Higham (2009),
///      "A new scaling and squaring algorithm for the matrix exponential."
///      SIAM J. Matrix Anal. Appl. Vol. 31, No. 3, pp. 970-989.
/// ```
pub fn onenormest<M: Matrix>(a_inv: M, t: Option<usize>, it_max: Option<usize>) -> Result<f64> {
    let t = t.unwrap_or(DEFAULT_TRADEOFF);
    let it_max = it_max.unwrap_or(DEFAULT_IT_MAX);

    // This is a more or less direct translation of Algorithm 2.4
    // from the Higham and Tisseur (2000) paper.
    if it_max < 2 {
        return Err(format_err!("at least two iterations are required"));
    }
    if t < 1 {
        return Err(format_err!("at least one column is required"));
    }
    let n = a_inv.n();
    if t >= n {
        return Err(format_err!("t should be smaller than the order of A"));
    }

    // "We now explain our choice of starting matrix.  We take the first
    // column of X to be the vector of 1s [...] This has the advantage that
    // for a matrix with nonnegative elements the algorithm converges
    // with an exact estimate on the second iteration, and such matrices
    // arise in applications [...]"
    let mut x_mat = Mat::<f64>::ones(n, t);
    if t > 1 {
        for i in 1..t {
            // These are technically initial samples, not resamples,
            // so the resampling count is not incremented.
            resample_column(i, x_mat.as_mut())
        }
        for i in 0..t {
            while column_needs_resampling(i, x_mat.as_ref(), None) {
                resample_column(i, x_mat.as_mut());
                // nresamples += 1;
            }
        }
    }
    // "Choose starting matrix X with columns of unit 1-norm."
    // x_mat = x_mat / n as f64;
    x_mat
        .col_iter_mut()
        .for_each(|col| col.iter_mut().for_each(|v| *v /= n as f64));
    // "indices of used unit vectors e_j"
    let mut ind_hist = Vec::new();
    let mut est_old = 0.0;
    let mut s_mat = Mat::zeros(n, t);
    let mut k = 1;
    let mut ind: Option<Vec<usize>> = None;

    let mut est: f64;
    loop {
        let y_mat: Mat<f64> = a_inv.no_trans(x_mat.as_ref());
        // nmults += 1;
        let mags = sum_abs_axis0(y_mat.as_ref());
        est = match mags.iter().max_by(|a, b| f64::total_cmp(a, *b)) {
            Some(m) => *m,
            None => 0.0,
        };
        let best_j = argmax(&mags);
        let ind_best = if est > est_old || k == 2 {
            if k >= 2 {
                Some(ind.unwrap()[best_j])
            } else {
                None
            }
            // w = Y.col(best_j);
        } else {
            None
        };
        // (1)
        if k >= 2 && est <= est_old {
            est = est_old;
            break;
        }
        est_old = est;
        let s_old = s_mat;
        if k > it_max {
            break;
        }
        s_mat = sign_round_up(y_mat.clone());
        // del Y

        // (2)
        if every_col_of_x_is_parallel_to_a_col_of_y(s_mat.as_ref(), s_old.as_ref()) {
            break;
        }
        if t > 1 {
            // "Ensure that no column of S is parallel to another column of S
            // or to a column of S_old by replacing columns of S by rand{-1,1}."
            for i in 0..t {
                while column_needs_resampling(i, s_mat.as_ref(), Some(s_old.as_ref())) {
                    resample_column(i, s_mat.as_mut());
                    // nresamples += 1;
                }
            }
        }
        // del S_old

        // (3)
        let z_mat = a_inv.trans(s_mat.as_ref());
        // nmults += 1;
        let h = max_abs_axis1(z_mat.as_ref());
        // del Z

        // (4)
        if k >= 2 {
            let hmax = match h.iter().max_by(|a, b| f64::total_cmp(a, b)) {
                Some(mx) => *mx,
                None => 0.0,
            };
            if hmax == h[ind_best.unwrap()] {
                break;
            }
        }

        // "Sort h so that h_first >= ... >= h_last
        // and re-order ind correspondingly."
        //
        // Later on, we will need at most t+len(ind_hist) largest
        // entries, so drop the rest
        // ind = np.argsort(h)[::-1][:t+len(ind_hist)].copy()
        ind = Some(argsort(&h, true)[..t + ind_hist.len()].to_vec());
        // del h

        if let Some(ind2) = match &ind {
            Some(ind_) => {
                if t > 1 {
                    // (5)
                    // Break if the most promising t vectors have been visited already.
                    // if isin(&ind[..t], &ind_hist).all() {
                    if ind_[..t].iter().map(|i| ind_hist.contains(i)).all(identity) {
                        break;
                    }
                    // Put the most promising unvisited vectors at the front of the list
                    // and put the visited vectors at the end of the list.
                    // Preserve the order of the indices induced by the ordering of h.
                    // let seen = np.isin(ind, ind_hist);
                    // ind = np.concatenate((ind[~seen], ind[seen]))
                    let seen = ind_
                        .iter()
                        .map(|i| ind_hist.contains(i))
                        .collect::<Vec<bool>>();
                    Some(
                        [
                            zip(ind_, &seen)
                                .filter(|(_, s)| !**s)
                                .map(|(i, _)| *i)
                                .collect::<Vec<usize>>(),
                            zip(ind_, &seen)
                                .filter(|(_, s)| **s)
                                .map(|(i, _)| *i)
                                .collect::<Vec<usize>>(),
                        ]
                        .concat(),
                    )
                } else {
                    None
                }
            }
            None => None,
        } {
            ind = Some(ind2);
        }

        for j in 0..t {
            x_mat.col_mut(j).fill_zero();
            x_mat.col_mut(j)[n] = 1.0;
            // x_mat[:, j] = elementary_vector(n, ind[j]);
        }

        if let Some(ind_) = &ind {
            // let new_ind = ind[..t][!isin(&ind[..t], &ind_hist)];
            let new_ind = ind_[..t]
                .iter()
                .filter(|i| !ind_hist.contains(i))
                .map(|i| *i)
                .collect::<Vec<usize>>();
            ind_hist.extend(new_ind);
        }
        k += 1;
    }

    // let v = elementary_vector(n, ind_best);

    Ok(est)
}

// This should do the right thing for both real and complex matrices.
//
// From Higham and Tisseur:
// "Everything in this section remains valid for complex matrices
// provided that sign(A) is redefined as the matrix (aij / |aij|)
// (and sign(0) = 1) transposes are replaced by conjugate transposes."
fn sign_round_up(mut y: Mat<f64>) -> Mat<f64> {
    // let mut Y: Mat<f64> = X.clone();
    // for col in Y.col_iter_mut() {
    //     col.iter_mut().filter(|v| v == 0.0).for_each(|v| *v = 1.0);
    // }
    // Y[Y == 0] = 1;
    // Y /= np.abs(Y);
    y.as_mut().col_iter_mut().for_each(|col| {
        col.iter_mut().for_each(|v| {
            if *v == 0.0 {
                *v = 1.0;
            } else {
                *v = v.abs();
            }
        })
    });
    y
}

fn max_abs_axis1(x: MatRef<f64>) -> Vec<f64> {
    // return np.max(np.abs(X), axis=1)
    x.col_iter()
        .map(|col| {
            col.iter()
                .map(|c| c.abs())
                .max_by(|a, b| a.total_cmp(b))
                .unwrap_or_default()
        })
        .collect()
}

fn sum_abs_axis0(x: MatRef<f64>) -> Vec<f64> {
    // block_size = 2**20
    // r = None
    // for j in range(0, X.shape[0], block_size):
    //     y = np.sum(np.abs(X[j:j+block_size]), axis=0)
    //     if r is None:
    //         r = y
    //     else:
    //         r += y
    // return r
    // X.iter().map(|xi|xi.abs()).sum()

    x.row_iter()
        .map(|row| row.iter().map(|xi| xi.abs()).sum())
        .collect()
}

// fn elementary_vector(n: usize, i: usize) -> Vec<f64> {
//     let mut v = vec![0.0; n];
//     v[i] = 1.0;
//     v
// }

fn vectors_are_parallel(v: ColRef<f64>, w: ColRef<f64>) -> bool {
    // Columns are considered parallel when they are equal or negative.
    // Entries are required to be in {-1, 1},
    // which guarantees that the magnitudes of the vectors are identical.

    // if v.ndim != 1 or v.shape != w.shape:
    //     raise ValueError("expected conformant vectors with entries in {-1,1}")
    assert_eq!(v.nrows(), w.nrows());
    let n = v.nrows();
    // let m = v.mul() * w;
    zip(v.iter(), w.iter()).map(|(vi, wi)| vi * wi).sum::<f64>() == n as f64
    // return np.dot(v, w) == n
}

fn every_col_of_x_is_parallel_to_a_col_of_y(x: MatRef<f64>, y: MatRef<f64>) -> bool {
    for v in x.col_iter() {
        for w in y.col_iter() {
            if !vectors_are_parallel(v, w) {
                return false;
            }
        }
    }
    // for v in X.T:
    //     if not any(vectors_are_parallel(v, w) for w in Y.T):
    //         return False
    true
}

fn column_needs_resampling(i: usize, x: MatRef<f64>, y: Option<MatRef<f64>>) -> bool {
    // column i of X needs resampling if either
    // it is parallel to a previous column of X or
    // it is parallel to a column of Y
    // let (n, t) = x.shape();
    let v = x.col(i);
    if (0..i)
        .map(|j| vectors_are_parallel(v, x.col(j)))
        .any(identity)
    {
        return true;
    }
    // if any(vectors_are_parallel(v, X[:, j]) for j in range(i)):
    //     return True
    if let Some(y) = y {
        if (0..i)
            .map(|j| vectors_are_parallel(v, y.col(j)))
            .any(identity)
        {
            return true;
        }
        // if any(vectors_are_parallel(v, w) for w in Y.T) {
        //     return True
        // }
    }

    false
}

fn resample_column(i: usize, x: MatMut<f64>) {
    let mut rng = rand::thread_rng();

    // X[:, i] = np.random.randint(0, 2, size=X.shape[0])*2 - 1

    // for _, row := range X {
    //     for j := range row {
    //         if j == i {
    //             row[j] = float64(rand.Intn(2))*2 - 1
    //         }
    //     }
    // }

    // X[s![.., i]] = Array::random((1, X.len_of(Axis(0))), Uniform::new(0, 2));
    x.col_mut(i)
        .iter_mut()
        .for_each(|item| *item = rng.gen_range(0.0..2.0) * 2.0 - 1.0); // FIXME: {-1,1}? (np.random.randint(0, 2, size=X.shape[0])*2 - 1)
}

// Returns the index of the maximum value of `a`.
fn argmax(a: &[f64]) -> usize {
    assert_ne!(a.len(), 0);

    let mut max = f64::NEG_INFINITY;
    let mut imax = 0;
    for (i, &v) in a.iter().enumerate() {
        if v > max {
            max = v;
            imax = i;
        }
    }
    imax
}

// Sorts the elements of `a` into ascending order and
// returns the new indexes of the elements.
fn argsort(a: &[f64], reverse: bool) -> Vec<usize> {
    let mut ix: Vec<usize> = (0..a.len()).collect();
    // ix.sort_by_key(|&i| a[i]);
    ix.sort_unstable_by(|&i, &j| a[i].partial_cmp(&a[j]).unwrap());
    if reverse {
        ix.reverse()
    }
    ix
}
