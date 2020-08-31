/// 贪心法相关

/// Jump Game
/// Given an array of non-negative integers, you are initially positioned at the first index of the array.
/// Each element in the array represents your maximum jump length at that position.
/// Determine if you are able to reach the last index.
/// For example:
/// A = [2,3,1,1,4], return true.
/// A = [3,2,1,0,4], return false.
pub fn can_jump(a: Vec<i32>) -> bool {
    let (mut i, mut reach) = (0, 1);
    while i < reach && reach < a.len() {
        reach = std::cmp::max(reach, i + 1 + (a[i] as usize));
        i += 1;
    }
    
    reach >= a.len()
}

/// Jump Game II
/// Given an array of non-negative integers, you are initially positioned at the first index of the array.
/// Each element in the array represents your maximum jump length at that position.
/// Your goal is to reach the last index in the minimum number of jumps.
/// For example: Given array A = [2,3,1,1,4]
/// e minimum number of jumps to reach the last index is 2. (Jump 1 step from index 0 to 1, then 3 steps
/// to the last index.)
pub fn jump(a: Vec<i32>) -> i32 {
    let (mut step, mut left, mut right) = (0, 0, 0);
    
    if a.len() <= 1 {
        return 0;
    }
    
    while left <= right {
        step += 1;
        let old_right = right;
        for i in left..=right {
            let new_right = i + (a[i] as usize);
            if new_right >= (a.len() - 1) {return step;}
            if new_right > right { right = new_right;}
        }
        left = old_right + 1;
    }
    0
}

/// Best Time to Buy and Sell Sto
/// Say you have an array for which the i-th element is the price of a given stock on day i.
/// If you were only permied to complete at most one transaction (ie, buy one and sell one share of the
/// stock), design an algorithm to find the maximum profit.
pub fn max_profit(prices: Vec<i32>) -> i32 {
    match prices.first() {
        Some(&x) => {
            prices.iter().skip(1).fold((0, x), |(profit, cur_min), &val| {
                (std::cmp::max(profit, val - cur_min), std::cmp::min(val, cur_min))
            }).0
        },
        None => 0,
    }
}

/// Best Time to Buy and Sell Stock II
/// Say you have an array for which the i-th element is the price of a given stock on day i.
/// Design an algorithm to find the maximum profit. You may complete as many transactions as you like
/// (ie, buy one and sell one share of the stock multiple times). However, you may not engage in multiple
/// transactions at the same time (ie, you must sell the stock before you buy again).
pub fn max_profit_ii(prices: Vec<i32>) -> i32 {
    match prices.first() {
        Some(&x) => {
            prices.iter().skip(1).fold((0, x), |(sum, prev), &val| {
                match val - prev {
                    y if y > 0 => (sum+y, val),
                    _ => (sum, val),
                }
            }).0
        },
        None => 0,
    }
}

/// Longest Substring Without Repeating Characters
/// Given a string, find the length of the longest substring without repeating characters. For example, the
/// longest substring without repeating leers for ”abcabcbb” is ”abc”, which the length is 3. For ”bbbbb”
/// the longest substring is ”b”, with the length of 1
pub fn length_of_longest_substr(s: String) -> i32 {
    let mut h = std::collections::HashMap::with_capacity(s.len());
    let (mut itr, itr_c) = (s.chars().enumerate().skip(0), s.chars().enumerate());
    let (mut cnt, mut max) = (0, 0);
    
    while let Some(x) = itr.next() {
        match h.get(&x.1) {
            Some(&idx) => {
                max = std::cmp::max(max, cnt);
                itr = itr_c.clone().skip(idx+1);
                cnt = 0;
                h.clear();
            },
            None => {
                h.insert(x.1, x.0);
                cnt += 1;
            },
        }
    }
    
    std::cmp::max(max, cnt) as i32
}

/// Container With Most Water
/// Given n non-negative integers a1; a2; :::; an, where each represents a point at coordinate (i; ai). n
/// vertical lines are drawn such that the two endpoints of line i is at (i; ai) and (i; 0). Find two lines, which
/// together with x-axis forms a container, such that the container contains the most water.
/// Note: You may not slant the container.
pub fn max_area(height: Vec<i32>) -> i32 {
    if height.len() <= 1 {
        return 0;
    }
    
    let (mut start, mut end, mut res) = (0, height.len() - 1, 0);
    while start < end {
        let area = std::cmp::min(height[start], height[end]) * ((end - start) as i32);
        res = std::cmp::max(res, area);
        if height[start] <= height[end] {
            start += 1;
        } else {
            end -= 1;
        }
    }
    res
}