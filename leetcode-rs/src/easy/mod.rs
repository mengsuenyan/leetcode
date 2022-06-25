use leetcode_meta::{Problem, Problems};
use std::sync::{Arc, RwLock};

pub mod p1;
pub mod p2;
pub mod p3;
pub mod p4;

lazy_static::lazy_static! {
    pub static ref PROBLEMS: Arc<RwLock<Problems<Problem>>> = {
        let p = Problems::new() | p1::PROBLEMS.read().unwrap().deref()
            | p2::PROBLEMS.read().unwrap().deref()
            | p3::PROBLEMS.read().unwrap().deref()
            | p4::PROBLEMS.read().unwrap().deref();
        Arc::new(RwLock::new(p))
    };
}
