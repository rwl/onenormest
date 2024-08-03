/// Returns the exact 1-norm of a sparse matrix in compressed-column format.
pub fn onenorm(n: usize, _row_ind: &[usize], col_ptr: &[usize], nz: &[f64]) -> f64 {
    let mut norm: f64 = 0.0;
    for i in 0..n {
        let col_start = col_ptr[i];
        let col_end = col_ptr[i + 1];

        let mut sum: f64 = 0.0;
        for j in col_start..col_end {
            sum += nz[j].abs();
        }

        if sum > norm {
            norm = sum
        }
    }
    norm
}
