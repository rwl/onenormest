use crate::Matrix;
use faer::perm::Perm;
use faer::sparse::linalg::lu::simplicial::SimplicialLu;
use faer::{Conj, Mat, Parallelism};

pub struct LU {
    row_perm: Perm<usize>,
    col_perm: Perm<usize>,
    lu: SimplicialLu<usize, f64>,
}

impl Matrix for LU {
    fn n(&self) -> usize {
        self.lu.ncols()
    }

    fn no_trans(&self, b: &full::Mat<f64>) -> full::Mat<f64> {
        let (rows, cols) = b.shape();

        let mut x = Mat::from_fn(rows, cols, |r, c| b[(r, c)]);
        let mut work = x.clone();

        self.lu.solve_in_place_with_conj(
            self.row_perm.as_ref(),
            self.col_perm.as_ref(),
            Conj::No,
            x.as_mut(),
            Parallelism::None,
            work.as_mut(),
        );

        full::Mat::from_fn(rows, cols, |r, c| x[(r, c)], true)
    }

    fn trans(&self, _x: &full::Mat<f64>) -> full::Mat<f64> {
        todo!()
    }
}
