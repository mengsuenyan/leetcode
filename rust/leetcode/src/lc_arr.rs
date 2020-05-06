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

/// Given a non-empty array of digits representing a non-negative integer, plus one to the integer.  
///   
/// The digits are stored such that the most significant digit is at the head of the list, and each element in the array contain a single digit.  
///   
/// You may assume the integer does not contain any leading zero, except the number 0 itself.  
///   
/// 来源：力扣（LeetCode）  
/// 链接：https://leetcode-cn.com/problems/plus-one  
/// 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。  
/// O(n)
pub fn plus_one(digits: Vec<i32>) -> Vec<i32> {
    let mut v = Vec::with_capacity(digits.len() + 1);
    
    let mut carry = 1;
    for &e in digits.iter().rev() {
        let ele = e + carry;
        carry = if ele > 9 {
            v.push(0);
            ele - 9
        } else {
            v.push(ele);
            0
        };
    }
    
    if carry > 0 {
        v.push(carry);
    }
    
    v.reverse();
    v
}

/// You are climbing a stair case. It takes n steps to reach to the top.  
///   
/// Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?  
///   
/// Note: Given n will be a positive integer.  
///   
/// Example 1:  
///   
/// 来源：力扣（LeetCode）  
/// 链接：https://leetcode-cn.com/problems/climbing-stairs  
/// 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。  
/// O(n)
pub fn climb_stairs(n: i32) -> i32 {
    let mut prev = 0;
    let mut cur = 1;
    for _ in 1..=n {
        let tmp = cur;
        cur = prev + cur;
        prev = tmp;
    }
    cur
}

/// Given a sorted array and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.  
///   
/// You may assume no duplicates in the array.  
///   
/// 来源：力扣（LeetCode）  
/// 链接：https://leetcode-cn.com/problems/search-insert-position  
/// 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。  
/// O( log(n) )
pub fn search_insert(nums: Vec<i32>, target: i32) -> i32 {
    match nums.binary_search(&target) {
        Ok(x) => x as i32,
        Err(x) => x as i32,
    }
}

/// Given an array nums and a value val, remove all instances of that value in-place and return the new length.  
///   
/// Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.  
///   
/// The order of elements can be changed. It doesn't matter what you leave beyond the new length.  
///   
/// 来源：力扣（LeetCode）  
/// 链接：https://leetcode-cn.com/problems/remove-element  
/// 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。  
/// T-O(n), M-O(1)
pub fn remove_element(nums: &mut Vec<i32>, val: i32) -> i32 {
    
    let mut cnt = 0;
    for i in 0..nums.len() {
        let ele = nums[i];
        if ele != val {
            nums[cnt] = ele;
            cnt += 1;
        }
    }
    
    cnt as i32
}

#[cfg(test)]
mod tests {
    #[test]
    fn three_sum() {
        let cases = [
            (vec![-1,0,1,2,-1,-4], vec![vec![-1,-1,2], vec![-1,0,1]]),
            (vec![], vec![]),
            (vec!["0",0,0], vec![vec![0,0,0]]),
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
    
    #[test]
    fn plus_one() {
        let cases = [
            (vec![1,2,3], vec![1,2,4]),
            (vec![9,9,9], vec![1,0,0,0]),
        ];
        
        for c in cases.iter() {
            assert_eq!(super::plus_one(c.0.clone()), c.1);
        }
    }
    
    #[test]
    fn climb_stairs() {
        let cases = [
            (3,3),
        ];

        for c in cases.iter() {
            assert_eq!(super::climb_stairs(c.0), c.1);
        }
    }
}