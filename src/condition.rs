use anyhow::{format_err, Result};

use crate::onenorm::onenorm;
use crate::onenormest::{onenormest, Matrix};

/// Estimates the 1-norm condition number given a sparse matrix and its inverse.
pub fn estimate(
    n: usize,
    row_ind: &[usize],
    col_ptr: &[usize],
    nz: &[f64],
    a_inv: impl Matrix,
    it_max: Option<usize>,
) -> Result<f64> {
    if n != a_inv.n() {
        return Err(format_err!(
            "matrix size {} must equal size of matrix inverse {}",
            n,
            a_inv.n()
        ));
    }

    let norm_a = onenorm(n, row_ind, col_ptr, nz);

    let norm_a_inv = onenormest(a_inv, None, it_max)?;

    Ok(norm_a * norm_a_inv)
}

/// Returns a cheap estimate of the reciprocal of the condition number
/// of a sparse matrix.
/// ```txt
///     min(abs(diag(A))) / max(abs(diag(A)))
/// ```
pub fn reciprocal(n: usize, row_ind: &[usize], col_ptr: &[usize], nz: &[f64]) -> f64 {
    let diag_a = csc_diag(n, row_ind, col_ptr, nz);

    let mut min_a: f64 = 0.0;
    let mut max_a: f64 = 0.0;

    for (i, u) in diag_a.iter().enumerate() {
        if *u == 0.0 || f64::is_nan(*u) {
            return 0.0;
        }
        let ui = u.abs();
        if i == 0 {
            min_a = ui;
        } else if ui.abs() < min_a {
            min_a = ui;
        }
        if i == 0 {
            max_a = ui;
        } else if ui.abs() > max_a {
            max_a = ui;
        }
    }

    if max_a == 0.0 {
        return 0.0;
    }
    let rc = min_a / max_a;

    if rc.is_nan() {
        0.0
    } else {
        rc
    }
}

fn csc_diag(n: usize, row_ind: &[usize], col_ptr: &[usize], nz: &[f64]) -> Vec<f64> {
    let mut diag = vec![0.0; n as usize];
    for i in 0..n {
        let col_start = col_ptr[i];
        let col_end = col_ptr[i + 1];

        let mut d: f64 = 0.0;
        for j in col_start..col_end {
            if row_ind[j] == i {
                d += nz[j];
            }
        }
        diag[i] = d;
    }
    diag
}
