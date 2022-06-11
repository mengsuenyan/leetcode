use leetcode_meta::{Problem, Problems};
use std::sync::{Arc, RwLock};

pub mod easy;
pub mod hard;
pub mod medium;

lazy_static::lazy_static! {
    pub static ref PROBLEMS: Arc<RwLock<Problems<Problem>>> = {
        let p = Problems::new() | easy::PROBLEMS.read().unwrap().deref()
            | medium::PROBLEMS.read().unwrap().deref()
            | hard::PROBLEMS.read().unwrap().deref();
        Arc::new(RwLock::new(p))
    };
}

pub mod prelude {
    pub use super::PROBLEMS;
    pub use leetcode_meta::prelude::*;
    pub use leetcode_pm::inject_description;
}
