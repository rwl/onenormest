use full::Mat;
use sparsetools::csc::CSC;
use sparsetools::{csc_matvecs, csr_matvecs};

use crate::Matrix;

pub struct CSCMatrix {
    csc: CSC<usize, f64>,
}

impl Matrix for CSCMatrix {
    fn n(&self) -> usize {
        self.csc.cols()
    }

    fn no_trans(&self, x: &Mat<f64>) -> Mat<f64> {
        let (n_row, n_col) = self.csc.shape();
        let n_vecs = x.cols();

        let mut y = Mat::zeros(n_row, n_vecs, true);
        csc_matvecs(
            n_row,
            n_col,
            n_vecs,
            self.csc.colptr(),
            self.csc.rowidx(),
            self.csc.values(),
            x.values(),
            y.values_mut(),
        );
        y
    }

    fn trans(&self, x: &Mat<f64>) -> Mat<f64> {
        let n_row = self.csc.rows();
        let n_col = self.csc.cols();
        let n_vecs = x.cols();

        let mut y = Mat::zeros(n_row, n_vecs, true);
        csr_matvecs(
            n_row,
            n_col,
            n_vecs,
            self.csc.colptr(),
            self.csc.rowidx(),
            self.csc.values(),
            x.values(),
            y.values_mut(),
        );
        y
    }
}

#[cfg(test)]
mod tests {
    use crate::matrix::sparsetools::CSCMatrix;
    use crate::onenormest;
    use sparsetools::csc::CSC;

    #[test]
    fn test_sparsetools() {
        let matrix = CSCMatrix {
            csc: CSC::from_dense(&[vec![1., 0., 0.], vec![5., 8., 2.], vec![0., -1., 0.]]),
        };
        let est = onenormest(matrix, None, None).unwrap();
        assert_eq!(est, 9.0);
    }
}
