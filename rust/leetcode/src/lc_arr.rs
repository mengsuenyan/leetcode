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
}