//! Sparse 1-norm and condition number estimator.

mod onenorm;
mod onenormest;

pub mod condition;
mod matrix;

pub use onenorm::onenorm;
pub use onenormest::*;
