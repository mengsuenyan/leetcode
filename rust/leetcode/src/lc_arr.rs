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

/// Given a set of distinct integers, nums, return all possible subsets (the power set).  
///   
/// Note: The solution set must not contain duplicate subsets.  
///   
/// 来源：力扣（LeetCode）  
/// 链接：https://leetcode-cn.com/problems/subsets  
/// 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。  
///  O(2^n)
pub fn subsets(nums: Vec<i32>) -> Vec<Vec<i32>> {
    let len = 2 << nums.len();
    let mut v = Vec::with_capacity(len);
    
    let mut sub = Vec::with_capacity(nums.len());
    for i in 0..(1i32 << (nums.len() as i32)) {
        for j in 0..(nums.len() as i32) {
            if (i & (1 << j)) != 0 {
                sub.push(nums[j as usize]);
            }
        }
        sub.sort();
        v.push(sub.clone());
        sub.clear();
    }
    
    v
}

/// Given a collection of integers that might contain duplicates, nums, return all possible subsets (the power set).  
/// Note: The solution set must not contain duplicate subsets.  
///   
/// 来源：力扣（LeetCode）  
/// 链接：https://leetcode-cn.com/problems/subsets-ii  
/// 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。  
/// O(2^n)
pub fn subsets_with_dup(nums: Vec<i32>) -> Vec<Vec<i32>> {
    let mut nums = subsets(nums);
    
    nums.sort();
    nums.dedup();
    nums
}

/// Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in-place.  
///   
/// 来源：力扣（LeetCode）  
/// 链接：https://leetcode-cn.com/problems/set-matrix-zeroes  
/// 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。  
/// O(n)
pub fn set_zeroes(matrix: &mut Vec<Vec<i32>>) {
    if matrix.is_empty() || matrix.first().unwrap().is_empty() {
        return;
    }
    let rows = matrix.len();
    let cols = matrix.first().unwrap().len();
    let mut zero_row = Vec::with_capacity(rows);
    let mut zero_col = Vec::with_capacity(cols);
    
    for (i, r) in matrix.iter().enumerate() {
        for (j, c) in r.iter().enumerate() {
            if *c == 0 {
                zero_col.push(j);
                zero_row.push(i);
            }
        }
    }
    
    for &i in zero_row.iter() {
        let len = matrix[i].len();
        matrix[i].clear();
        matrix[i].resize(len, 0);
    }
    
    for &j in zero_col.iter() {
        for r in matrix.iter_mut() {
            r[j] = 0;
        }
    }
}

/// Single Number II
/// 
/// Given an array of integers, every element appears three times except for one. Find that single one.
/// Note: Your algorithm should have a linear runtime complexity. Could you implement it without using
/// extra memory?
pub fn single_number_ii(arrs: Vec<i32>) -> i32 {
    // 记每个数为$n_{i,j}$, 其中i表示第i个数(共有N个数), j表示第i个数的第j位(每个数有B位). 记要求的唯一的那个数为$x$,
    // 那么有$x_j=(\sum_{i=0}^{N} n_{i,j}) \mod 3$
    // tc: O(n), sc: O(1)
    let (mut one, mut two, mut three) = (0, 0, 0);
    arrs.iter().for_each(|&ele| {
        two |= one & ele;
        one ^= ele;
        three = !(one & two);
        one &= three;
        two &= three;
    });
    
    one
}

/// Single Number
/// 
/// Given an array of integers, every element appears twice except for one. Find that single one.
/// Note: Your algorithm should have a linear runtime complexity. Could you implement it without using
/// extra memory?
pub fn single_number(arrs: Vec<i32>) -> i32 {
    // tc: O(n), sc: O(1)
    let mut one = 0;
    arrs.iter().for_each(|&ele| {
        one ^= ele;
    });
    one
}

/// Candy
/// 
/// There are N children standing in a line. Each child is assigned a rating value.
/// You are giving candies to these children subjected to the following requirements:
/// - Each child must have at least one candy.
/// - Children with a higher rating get more candies than their neighbors.
/// What is the minimum candies you must give?
pub fn candy(ratings: Vec<i32>) -> i32 {
    // tc: O(n), sc: O(n)
    if ratings.is_empty() {
        return 0;
    }
    
    let mut candys = Vec::with_capacity(ratings.len());
    let mut need_candy = 1;
    let (mut prev, mut next) = (match ratings.first() { 
        Some(&x) =>  x, None => std::i32::MIN,
    }, match ratings.last() {
        Some(&x) => x, None => std::i32::MIN,
    });
    
    candys.resize(ratings.len(), 1);
    ratings.iter().skip(1).zip(candys.iter_mut().skip(1)).for_each(|(&cur, candy)| {
        if cur > prev {
            need_candy += 1;
            *candy = std::cmp::max(need_candy, *candy);
        } else { 
            need_candy = 1;
        }
        prev = cur;
    });
    
    need_candy = 1;
    ratings.iter().rev().skip(1).zip(candys.iter_mut().rev().skip(1)).for_each(|(&cur, candy)| {
        if cur > next {
            need_candy += 1;
            *candy = std::cmp::max(need_candy, *candy);
        } else {
            need_candy = 1;
        }
        next = cur;
    });
    
    candys.iter().sum()
}

/// Gas Station
/// 
/// There are N gas stations along a circular route, where the amount of gas at station i is gas[i].
/// You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from station i to its
/// next station (i+1). You begin the journey with an empty tank at one of the gas stations.
/// Return the starting gas station’s index if you can travel around the circuit once, otherwise return -1.
/// Note: The solution is guaranteed to be unique
pub fn can_complete_circuit(gas: Vec<i32>, cost: Vec<i32>) -> i32 {
    // 能走完, 则必然$\sum_{i} gas \gt \sum_i cost$. 
    // 又从一站前往下一站, 则必然当前气量大于消耗气量.
    // tc: O(n), sc: O(1)
    debug_assert_eq!(gas.len(), cost.len(), "The number of gas must equal to the number of cost");
    
    let (mut total, mut sum, mut idx, mut i) = (0, 0, -1, 0);
    gas.iter().zip(cost.iter()).for_each(|(&g, &c)| {
        let s = g - c;
        sum += s;
        total += s;
        if sum < 0 {
            idx = i;
            sum = 0;
        }
        i += 1;
    });
    
    if total >= 0 {idx+1} else {-1}
}

/// Gray Coode
/// 
/// The gray code is a binary numeral system where two successive values differ in only one bit.
/// Given a non-negative integer n representing the total number of bits in the code, print the sequence
/// of gray code. A gray code sequence must begin with 0.
/// For example, given n = 2, return [0,1,3,2]. Its gray code sequence is:
/// 00 - 0
/// 01 - 1
/// 11 - 3
/// 10 - 2
/// Note:
/// - For a given n, a gray code sequence is not uniquely defined.
/// - For example, [0,2,3,1] is also a valid gray code sequence according to the above definition.
/// - For now, the judge is able to judge based on one instance of gray code sequence. Sorry about that.
pub fn gray_code(n: i32) -> Vec<i32> {
    // 格雷码序列可按照如下方法生成, 记格雷码序列(共N个格雷码)的第n个格雷码为$g$, 那么有$g_0=n_0, g_i=n_i \oplus n_{i-1}$.
    // 其中下标$i$表示该数的第$N-i-1$位. 补充: 其逆运算为$n_0=g_0, n_i=g_i \oplus n_{i-1}$
    // tc: O(2^n), sc: O(1)
    if n >= 0 {
        let n = 1 << n;
        let mut res = Vec::with_capacity(n as usize);
        for i in 0..n {
            res.push(i ^ (i >> 1));
        }
        
        res
    } else {
        Vec::new()
    }
}

/// Rotate Image
/// 
/// You are given an n × n 2D matrix representing an image.
/// Rotate the image by 90 degrees (clockwise).
/// Follow up: Could you do this in-place?
pub fn rotate(matrix: &mut Vec<Vec<i32>>) {
    // 沿水平线翻转一次, 然后沿主对角线翻转一次;
    // tc: O(n^2), sc: O(n)
    let n = matrix.len();
    for i in 0..(n >> 1) {
        matrix.swap(i, n - 1 - i);
    }
    
    let p = matrix.as_mut_ptr();
    unsafe {
        for i in 0..n {
            let up = p.add(i) ;
            for j in (i+1)..n {
                let down = p.add(j);
                std::ptr::swap((*up).as_mut_ptr().add(j), (*down).as_mut_ptr().add(i));
            }
        }
    }
}

/// Trapping Rain Water
/// 
/// Given n non-negative integers representing an elevation map where the width of each bar is 1, compute
/// how much water it is able to trap aer raining.
/// For example, Given [0,1,0,2,1,0,1,3,2,1,2,1], return 6.
pub fn trap(a: Vec<i32>) -> i32 {
    // 木桶效应
    // tc: O(n), sc: O(1)

    match a.iter().enumerate().fold((0, None), |res, (idx, &x)| {
        if x > res.0 {
            (x, Some(idx))
        } else {
            (res.0, res.1)
        }
    }) {
        (_, Some(max)) => {
            let (mut water, mut tmp) = (0, 0);
            a.iter().take(max).for_each(|&x| {
                if x > tmp {
                    tmp = x;
                } else {
                    water += tmp - x;
                }
            });
            
            tmp = 0;
            a.iter().rev().take(a.len() - 1 - max).for_each(|&x| {
                if x > tmp {
                    tmp = x;
                } else {
                    water += tmp - x;
                }
            });
            
            water
        },
        _ => 0,
    }
}

/// Valid Sudoku
/// 
/// Determine if a Sudoku is valid, according to: Sudoku Puzzles - The Rules
/// http://sudoku.com.au/TheRules.aspx .
/// The Sudoku board could be partially filled, where empty cells are filled with the character '.'.
pub fn is_valid_sudoku(board: Vec<Vec<char>>) -> bool {
    // tc: O(n^2), sc: O(1)
    debug_assert_eq!(board.len(), 9, "The Sudoku size must be 9 * 9");
    debug_assert!({let mut a = true; for ele in board.iter() {if ele.len() != 9 {a = false; break;}} a}, "The Sudoku size must be 9 * 9");
    
    let check = |ch: char, is_used: &mut [bool;9]| {
        if ch == '.' { true }
        else {
            let idx = (ch as u32 - '1' as u32) as usize;
            if is_used[idx] { false }
            else {
                is_used[idx] = true;
                true
            }
        }    
    };
    
    let p = board.as_ptr();
    unsafe {
        // 行,列检查
        for i in 0..9 {
            let mut usedr = [false; 9];
            let mut usedc = [false; 9];
            let row = (*(p.add(i))).as_ptr();
            for j in 0..9 {
                let ele = *row.add(j);
                if !check(ele, &mut usedr) {
                    return false;
                }

                let col = (*(p.add(j))).as_ptr();
                let ele = *col.add(i);
                if !check(ele, &mut usedc) {
                    return false;
                }
            }
        }
        
        // 子格检查
        for r in 0..3 {
            for c in 0..3 {
                let mut used = [false; 9];
                for i in (r*3)..(r*3 + 3) {
                    let row = (*(p.add(i))).as_ptr();
                    for j in (c*3)..(c*3+3) {
                        let ele = *row.add(j);
                        if !check(ele, &mut used) {
                            return false;
                        }
                    }
                }
            }
        }
    }
    
    true
}

/// Next Permutation
/// 
/// Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.
/// If such arrangement is not possible, it must rearrange it as the lowest possible order (ie, sorted in
/// ascending order).
/// The replacement must be in-place, do not allocate extra memory.
/// Here are some examples. Inputs are in the left-hand column and its corresponding outputs are in the
/// right-hand column.
/// 1,2,3 → 1,3,2
/// 3,2,1 → 1,2,3
/// 1,1,5 → 1,5,1
pub fn next_permutation(num: &mut Vec<i32>) {
    // http://fisherlei.blogspot.com/2012/12/leetcode-next-permutation.html
    // 按字典序找下一个置换
    // 从右往左, 找出第一个不按增序递增的数, 记为a, 位置记为ia;
    // 从右往左, 找出第一个大于a的数, 记为b;
    // 替换a和b;
    // 逆序ia位置之后的数;
    // tc: O(n), sc: O(1)
    if num.is_empty() {return;}
    
    let (mut p1, mut last) = (0, std::i32::MIN);
    for (idx, &ele) in num.iter().rev().enumerate() {
        if ele < last {
            p1 = idx;
            last = ele;
            break;
        } else {
            last = ele;
        }
    }
    
    if p1 == 0 {
        return num.reverse();
    }
    
    let mut p2 = 0;
    for (idx, &ele) in num.iter().rev().enumerate() {
        p2 = idx;
        if ele > last {
            break;
        }
    }
    
    let (p1, p2) = (num.len() - p1 - 1, num.len() - p2 - 1);
    num.swap(p1, p2);
    num[(p1+1)..].reverse();
}

/// Permutation Sequence
/// 
/// Thee set [1,2,3,…,n] contains a total of n! unique permutations.
/// By listing and labeling all of the permutations in order, We get the following sequence (ie, for n = 3):
/// "123"
/// "132"
/// "213"
/// "231"
/// "312"
/// "321"
/// Given n and k, return the kth permutation sequence.
/// Note: Given n will be between 1 and 9 inclusive
pub fn get_permutation(n: i32, k: i32) -> String {
    // 康托编码, 记第k个置换为$a_1 a_2 \dots a_n$, 那么:
    // $a_1  = k / (n-1)!, k_2 = k % (n-1), a_2 = k_2 / (n-2)!$, 依次归纳;
    // tc: O(n), sc: O(1)
    debug_assert!(n >= 1 && n <= 9, "n must be in the range of [1,9]");
    debug_assert!(k >= 1 && k <= (1..=n).fold(1, |m, x|{x*m}), "k must be in the range of [1,{}!]", n);
    
    let (mut seq, mut res) = (String::with_capacity(n as usize), String::with_capacity(n as usize));
    (1..=(n as u8)).for_each(|x| {seq.push((b'0' + x) as char);});
    let (mut base, mut k) = ((1..=(n-1)).fold(1, |m, x|{x*m}), k - 1);
    
    for i in (1..=(n-1)).rev() {
        let pos = (k / base) as usize;
        let c = seq.chars().nth(pos);
        res.push(c.unwrap());
        seq.remove(pos);
        k %= base;
        base /= i;
    }
    
    res.push(seq.chars().nth(0).unwrap());
    
    res
}

/// 3Sum Closet
/// 
/// Given an array S of n integers, find three integers in S such that the sum is closest to a given number,
/// target. Return the sum of the three integers. You may assume that each input would have exactly one
/// solution.
/// For example, given array S = {-1 2 1 -4}, and target = 1.
/// The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
pub fn three_sum_closet(num: Vec<i32>, target: i32) -> i32 {
    // 先排序, 再左右夹逼
    // tc: O(n^2), sc: O(1)
    debug_assert!(num.len() > 2, "the size of S must be great to 2");
    let (mut num, mut res, mut min_gap) = (num, 0, std::i32::MAX);
    num.sort();
    
    num.iter().take(num.len() - 1).enumerate().for_each(|(idx, &a)| {
        let (mut b_itr, mut c_itr) = 
            (num.iter().skip(idx + 1).enumerate().peekable(), num.iter().rev().enumerate().peekable());
        let (mut b, mut c) = (*b_itr.peek().unwrap(), *c_itr.peek().unwrap());
        while (b.0 + idx + 1) < (num.len() - 1 - c.0){
            let sum =  a + *b.1 + *c.1;
            let gap = (sum - target).abs();
            if gap < min_gap {
                res = sum;
                min_gap = gap;
            }
            if sum < target {
                b = b_itr.next().unwrap();
            } else {
                c = c_itr.next().unwrap();
            }
        }
    });
    
    res
}

/// Two Sum
/// 
/// Given an array of integers, find two numbers such that they add up to a specific target number.
/// The function twoSum should return indices of the two numbers such that they add up to the target
/// 
/// You may assume that each input would have exactly one solution.
pub fn two_sum(num: Vec<i32>, target: i32) -> Vec<i32> {
    // tc: O(n), sc: O(n)
    let mut res = Vec::with_capacity(2);
    let mut h = std::collections::HashMap::with_capacity(num.len());
    
    for (idx, ele) in num.iter().enumerate() {
        if let Some(&x) = h.get(ele) {
            res.push(x as i32);
            res.push(idx as i32);
            break;
        } else {
            h.insert(target - *ele, idx);
        }
    }
    
    res
}

/// Longest Consecutive Sequence
/// 
/// Given an unsorted array of integers, find the length of the longest consecutive elements sequence.
/// For example, Given [100, 4, 200, 1, 3, 2], The longest consecutive elements sequence is [1,
/// 2, 3, 4]. Return its length: 4.
/// Your algorithm should run in O(n) complexity
pub fn longest_consecutive(num: Vec<i32>) -> i32 {
    // sc: O(n), tc: O(n)
    let mut h = std::collections::HashSet::with_capacity(num.len());
    num.iter().for_each(|&x| {
        h.insert(x);
    });
    
    h.iter().fold(0, |longest, &x| {
        let (mut len, mut nxt) = (1, x - 1);
        // 从连续序列的头开始查找, 防止某些子序列遍历多次
        if !h.contains(&nxt) {
            nxt += 2;
            while h.contains(&nxt) {
                len += 1;
                nxt += 1;
            }
        }
        std::cmp::max(longest, len)
    })
}

/// Median of Two Sorted Arrays
/// 
/// There are two sorted arrays A and B of size m and n respectively. Find the median of the two sorted
/// arrays. The overall run time complexity should be O(log(m + n))
pub fn find_median_sorted_array(a: Vec<i32>, b: Vec<i32>) -> f64 {
    let mut c = Vec::with_capacity(a.len() + b.len());
    let (mut a_itr, mut b_itr) = (a.iter().peekable(), b.iter().peekable());
    
    loop {
        match (a_itr.peek(), b_itr.peek())  {
            (Some(&&x), Some(&&y)) => {
                if x < y { c.push(x); a_itr.next(); }
                else if x > y {c.push(y); b_itr.next();}
                else {c.push(x); c.push(y); a_itr.next(); b_itr.next();};
            },
            (Some(&&x), None) => {
                c.push(x);
                a_itr.next();
            },
            (None, Some(&&y)) => {
                c.push(y);
                b_itr.next();
            }
            _ => {break;},
        }
    }

    let len = a.len() + b.len();
    if (len & 0x1) == 1 {
        c[len >> 1] as f64
    } else {
        if len == 0 {
            0f64
        } else {
            (c[len >> 1] as f64 + c[(len - 1) >> 1] as f64) / 2f64
        }
    }
}

/// Remove Duplicates from Sorted Array 
/// 
/// Given a sorted array, remove the duplicates in place such that each element appear only
/// once and return the new length.
/// Do not allocate extra space for another array, you must do this in place with constant memory.
/// For example, Given input array A = [1,1,2],
/// Your function should return length = 2, and A is now [1,2].
pub fn remove_duplicates(a: &mut Vec<i32>) -> i32 {
    a.dedup();
    a.len() as i32
}

/// Remove Duplicates from Sorted Array II
/// 
/// Follow up for ”Remove Duplicates”: What if duplicates are allowed at most twice?
/// For example, Given sorted array A = [1,1,1,2,2,3],
/// Your function should return length = 5, and A is now [1,1,2,2,3]
pub fn remove_duplicates_ii(a: &mut Vec<i32>) -> i32 {
    let mut cnt = 1;
    a.dedup_by(|a, b|->bool{
        cnt = if a == b {cnt + 1} else {1};
        if cnt > 2 { true } else { false }
    });

    a.len() as i32
}

/// Search in Rotated Sorted Array
/// 
/// Suppose a sorted array is rotated at some pivot unknown to you beforehand.
/// (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
/// You are given a target value to search. If found in the array return its index, otherwise return -1.
/// You may assume no duplicate exists in the array.
pub fn search(nums: Vec<i32>, target: i32) -> i32 {
    let mut first = 0;
    let mut last = nums.len();
    let mut idx = -1;

    while first != last {
        let mid = (first + last) >> 1;
        if nums[mid] == target {
            idx = mid as i32;
            break;
        } else if nums[first] <= nums[mid] {
            if nums[first] <= target && target < nums[mid] {
                last = mid;
            } else {
                first = mid + 1;
            }
        } else {
            if nums[mid] < target && target <= nums[last - 1] {
                first = mid + 1;
            } else {
                last = mid;
            }
        }
    }

    idx
}

/// Search in Rotated Sorted Array II
/// 
/// Follow up for ”Search in Rotated Sorted Array”: What if duplicates are allowed?
/// Would this affect the run-time complexity? How and why?
/// Write a function to determine if a given target is in the array
pub fn search_ii(nums: Vec<i32>, target: i32) -> bool {
    let mut first = 0;
    let mut last = nums.len();
    let mut isexist = false;

    while first != last {
        let mid = (first + last) >> 1;
        if nums[mid] == target {
            isexist = true;
            break;
        } else if nums[first] < nums[mid] {
            if nums[first] <= target && target < nums[mid] {
                last = mid;
            } else {
                first = mid + 1;
            }
        } else if nums[first] > nums[mid] {
            if nums[mid] <= target && target <= nums[last - 1] {
                first = mid + 1;
            } else {
                last = mid;
            }
        } else {
            first += 1;
        }
    }

    isexist
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
    
    #[test]
    fn subsets() {
        let cases = [
            ([1,2,3], [
                vec![3],
                vec![1],
                vec![2],
                vec![1,2,3],
                vec![1,3],
                vec![2,3],
                vec![1,2],
                vec![]
            ]),
        ];
        for c in cases.iter() {
            let mut s = super::subsets(c.0.to_vec());
            s.sort();
            let mut c1 = c.1.to_vec();
            c1.sort();
            assert_eq!(s, c1);
        }
    }
    
    #[test]
    fn single_number_ii() {
        let cases = [
            (3, vec![2,2,3,2]),
            (99, vec![0,1,0,1,0,1,99]),
        ];
        
        cases.iter().for_each(|x| {
            assert_eq!(x.0, super::single_number_ii(x.1.to_vec()));
        });
    }
    
    #[test]
    fn single_number() {
        let cases = [
            (1, vec![2,2,1]),
            (4, vec![4,1,2,1,2]),
        ];
        cases.iter().for_each(|x| {
            assert_eq!(x.0, super::single_number(x.1.to_vec()), "case: {:?}", x.0);
        });
    }
    
    #[test] 
    fn candy() {
        let cases = [
            (5, vec![1,0,2]),
            (4, vec![1,2,2]),
        ];
        cases.iter().for_each(|x| {
            assert_eq!(x.0, super::candy(x.1.to_vec()), "case: {:?}", x.0);
        });
    }
    
    #[test]
    fn can_complete_circuit() {
        let cases = [
            (3, vec![1,2,3,4,5], vec![3,4,5,1,2]),
        ];
        cases.iter().for_each(|x| {
            assert_eq!(x.0, super::can_complete_circuit(x.1.to_vec(), x.2.to_vec()), "case: {:?}", x.0);
        });
    }
    
    #[test]
    fn gray_code() {

        let cases = [
            (vec![0,1,3,2], 2),
            (vec![0], 0),
        ];
        cases.iter().for_each(|x| {
            assert_eq!(x.0, super::gray_code(x.1), "case: {:?}", x.0);
        });
    }
    
    #[test]
    fn rotate() {

        let mut cases = [
            (vec![vec![7,4,1],vec![8,5,2],vec![9,6,3]], vec![vec![1,2,3],vec![4,5,6],vec![7,8,9]]),
            (vec![vec![15,13,2,5],vec![14,3,4,1],vec![12,6,8,9], vec![16,7,10,11]],
            vec![vec![5,1,9,11],vec![2,4,8,10],vec![13,3,6,7],vec![15,14,12,16]]),
        ];
        cases.iter_mut().for_each(|x| {
            super::rotate(&mut x.1);
            assert_eq!(x.0, x.1, "case: {:?}", x.0);
        });
    }
    
    #[test]
    fn trap() {

        let cases = [
            (6,vec![0,1,0,2,1,0,1,3,2,1,2,1]),
        ];
        cases.iter().for_each(|x| {
            assert_eq!(x.0, super::trap(x.1.to_vec()), "case: {:?}", x.0);
        });
    }
    
    #[test]
    fn is_valid_sudoku() {

        let cases = [
            (true,
             vec![
                 vec!['5','3','.','.','7','.','.','.','.'],
                 vec!['6','.','.','1','9','5','.','.','.'],
                 vec!['.','9','8','.','.','.','.','6','.'],
                 vec!['8','.','.','.','6','.','.','.','3'],
                 vec!['4','.','.','8','.','3','.','.','1'],
                 vec!['7','.','.','.','2','.','.','.','6'],
                 vec!['.','6','.','.','.','.','2','8','.'],
                 vec!['.','.','.','4','1','9','.','.','5'],
                 vec!['.','.','.','.','8','.','.','7','9']
             ]
            ),
            (false,
             vec![
                 vec!['8','3','.','.','7','.','.','.','.'],
                 vec!['6','.','.','1','9','5','.','.','.'],
                 vec!['.','9','8','.','.','.','.','6','.'],
                 vec!['8','.','.','.','6','.','.','.','3'],
                 vec!['4','.','.','8','.','3','.','.','1'],
                 vec!['7','.','.','.','2','.','.','.','6'],
                 vec!['.','6','.','.','.','.','2','8','.'],
                 vec!['.','.','.','4','1','9','.','.','5'],
                 vec!['.','.','.','.','8','.','.','7','9']
             ]
            ),
        ];
        cases.iter().for_each(|x| {
            assert_eq!(x.0, super::is_valid_sudoku(x.1.to_vec()), "case: {:?}", x.0);
        });
    }
    
    #[test]
    fn next_permutation() {
        let mut cases = [
            // (vec![1,3,2], vec![1,2,3]),
            // (vec![1,2,3], vec![3,2,1]),
            // (vec![1,5,1], vec![1,1,5]),
            (vec![1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100],
                vec![100,99,98,97,96,95,94,93,92,91,90,89,88,87,86,85,84,83,82,81,80,79,78,77,76,75,74,73,72,71,70,69,68,67,66,65,64,63,62,61,60,59,58,57,56,55,54,53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]),
        ];
        cases.iter_mut().for_each(|x| {
            super::next_permutation(&mut x.1);
            assert_eq!(x.0, x.1, "case: {:?}", x.1);
        });
    }
    
    #[test]
    fn get_permutation() {

        let cases = [
            ((3,3), "213"),
            ((4,9), "2314"),
        ];
        cases.iter().for_each(|x| {
            assert_eq!(x.1, super::get_permutation((x.0).0, (x.0).1).as_str(), "case: {:?}", x.0);
        });
    }
    
    #[test]
    fn three_sum_closet() {

        let cases = [
            ((vec![-1,2,1,-4],1),2),
            ((vec![1,1,-1],0),1)
        ];
        cases.iter().for_each(|x| {
            assert_eq!(x.1, super::three_sum_closet((x.0).0.to_vec(), (x.0).1), "case: {:?}", x.0);
        });
    }
    
    #[test]
    fn two_sum() {

        let cases = [
            ((vec![2,7,11,15],9),vec![0,1]),
            ((vec![3,2,4],6),vec![1,2]),
        ];
        cases.iter().for_each(|x| {
            assert_eq!(x.1, super::two_sum((x.0).0.to_vec(), (x.0).1), "case: {:?}", x.0);
        });
    }
    
    #[test]
    fn longest_consecutive() {

        let cases = [
            (vec![100,4,200,1,3,2], 4),
        ];
        cases.iter().for_each(|x| {
            assert_eq!(x.1, super::longest_consecutive(x.0.to_vec()), "case: {:?}", x.0);
        });
    }
    
    #[test]
    fn find_median_sorted_array() {

        let cases = [
            ((vec![1,3],vec![2]),2.0),
            ((vec![1,2],vec![3,4]),2.5),
        ];
        cases.iter().for_each(|x| {
            assert_eq!(x.1, super::find_median_sorted_array((x.0).0.to_vec(), (x.0).1.to_vec()), "case: {:?}", x.0);
        });
    }
    
    #[test]
    fn search() {

        let cases = [
            ((vec![4,5,6,7,0,1,2], 0),4),
        ];
        cases.iter().for_each(|x| {
            assert_eq!(x.1, super::search((x.0).0.to_vec(), (x.0).1), "case: {:?}", x.0);
        });
    }
}