use leetcode_meta::{Problem, Problems};
use std::sync::{Arc, RwLock};

pub mod p1;
pub mod p2;

lazy_static::lazy_static! {
    pub static ref PROBLEMS: Arc<RwLock<Problems<Problem>>> = {
        let p = Problems::new() | p1::PROBLEMS.read().unwrap().deref()
            | p2::PROBLEMS.read().unwrap().deref();
        Arc::new(RwLock::new(p))
    };
}
