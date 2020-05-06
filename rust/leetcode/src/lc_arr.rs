use std::collections::{HashMap, HashSet}; 

macro_rules! impl_three_sum {
    ($Neg:ident, $Pos:ident, $V:ident, $Tgt: ident) => {
        for (i, &n1) in $Neg.iter().enumerate() {
            let rem_numc = &$Neg[(i+1)..];
            for &n2 in rem_numc.iter() {
                let n3 = $Tgt - n1 - n2;
                if $Pos.binary_search(&n3).is_ok() {
                    let mut ele = vec![n1,n2,n3];
                    ele.sort();
                    $V.push(ele);
                }
            }
        }
    };
}

pub fn three_sum(nums: Vec<i32>) -> Vec<Vec<i32>> {
    const TGT: i32 = 0;
    let mut v: Vec<Vec<i32>> = Vec::new();
   
    let mut numc: Vec<i32> = nums.clone();
    numc.sort();
    numc.dedup_by(|a,b| *a == 0 && *b == 0);

    let (zero_idx, is_invalid) = match numc.binary_search(&TGT) { 
        Ok(i) => { if nums.len() - numc.len() > 1 { v.push(vec![0,0,0]); }; (i + 1, false)},
        Err(i) => if i == numc.len() { (i, true) } else { (i, false) },
    };
    
    if is_invalid {
        return v;
    }

    let numc = numc.as_mut_slice();
    let neg_numc = &numc[0..zero_idx];
    let pos_numc = &numc[zero_idx..];
    
    impl_three_sum!(neg_numc, pos_numc, v, TGT);
    impl_three_sum!(pos_numc, neg_numc, v, TGT);
    v.sort();
    v.dedup();
    v
}

pub fn four_sum(nums: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
    let mut v: Vec<Vec<i32>> = Vec::new();
    if nums.len() < 4 {
        return v;
    }
    
    v.sort_unstable();
    let numc = nums.as_slice();
    let mut cache : HashMap<i32, HashSet<(usize, usize, i32, i32)>> = HashMap::new();
    for (i, &e1) in numc.iter().enumerate() {
        let rem_numc = &numc[(i+1)..];
        for (j, &e2) in rem_numc.iter().enumerate() {
            let r = e1 + e2;
            match cache.get_mut(&r) { 
                Some(s) => {
                    s.insert((i, j+1+i, e1, e2));
                },
                None => {
                    let mut s = HashSet::new();
                    s.insert((i, j+1+i, e1, e2));
                    cache.insert(r, s);
                },
            };
        }
    }
    
    for (i, &e1) in numc.iter().enumerate() {
        let rem_numc = &numc[(i+1)..];
        for (j, &e2) in rem_numc.iter().enumerate() {
            let r = target - e1 - e2;
            match cache.get_mut(&r) {
                Some(s) => {
                    for ele in s.iter() {
                        if ele.0 != i && ele.1 != (j+i+1) && ele.0 != (j+i+1) && ele.1 != i {
                            let mut t = vec![e1, e2, ele.2, ele.3];
                            t.sort();
                            v.push(t);
                        }
                    }
                },
                None => {},
            }
        }
    }
    
    v.sort();
    v.dedup();
    v
}

#[cfg(test)]
mod tests {
    #[test]
    fn three_sum() {
        let cases = [
            (vec![-1,0,1,2,-1,-4], vec![vec![-1,-1,2], vec![-1,0,1]]),
            (vec![], vec![]),
            (vec![0,0,0], vec![vec![0,0,0]]),
            (vec![1,1,-2], vec![vec![-2,1,1]]),
        ];
        
        for ele in cases.iter() {
            assert_eq!(super::three_sum(ele.0.clone()), ele.1.clone());
        }
    }
    
    #[test]
    fn four_sum() {
        let cases = [
            (0, vec![1,0,-1,0,-2,2], vec![vec![-2,-1,1,2], vec![-2,0,0,2], vec![-1,0,0,1]])
        ];
        
        for ele in cases.iter() {
            assert_eq!(super::four_sum(ele.1.clone(), ele.0), ele.2.clone());
        }
    }
}