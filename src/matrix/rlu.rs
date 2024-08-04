use crate::Matrix;
use full::Mat;
use std::iter::zip;

pub struct RLU {
    n: usize,
    col_perm: Vec<usize>,
    l_mat: rlu::Matrix<usize, f64>,
    u_mat: rlu::Matrix<usize, f64>,
    row_perm: Vec<Option<usize>>,
    // row_perm_inv: Vec<usize>,
    // reachable: Vec<Vec<usize>>,
}

impl Matrix for RLU {
    fn n(&self) -> usize {
        self.n
    }

    fn no_trans(&self, x_mat: &Mat<f64>) -> Mat<f64> {
        assert!(x_mat.col_major());

        let mut b = vec![0.0; x_mat.values().len()];

        for (x_col, b_col) in zip(
            x_mat.values().chunks_exact(x_mat.rows()),
            b.chunks_exact_mut(x_mat.rows()),
        ) {
            // for x_col in x_mat.col_iter() {
            let mut x = vec![0.0; self.n];
            for i in 0..self.n {
                x[self.row_perm[i].unwrap()] = x_col[i];
            }

            rlu::lsolve(&self.l_mat, &mut x);
            rlu::usolve(&self.u_mat, &mut x);

            // rlu::utsolve(&self.u_mat, &mut x);
            // rlu::ltsolve(&self.l_mat, &mut x);

            // let mut b = vec![0.0; self.n];
            for i in 0..self.n {
                b_col[self.col_perm[i]] = x[i];
            }
        }

        Mat::new(self.n, x_mat.cols(), b, true)
    }

    fn trans(&self, _x: &Mat<f64>) -> Mat<f64> {
        todo!()
    }
}
