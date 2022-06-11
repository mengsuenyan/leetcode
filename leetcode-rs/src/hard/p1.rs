use crate::easy::p1::ListNode;
use crate::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::{Arc, RwLock};

lazy_static::lazy_static! {
    pub static ref PROBLEMS: Arc<RwLock<Problems<Problem>>> =Arc::new(RwLock::new(Problems::new()));
}

#[inject_description(
    problems = "PROBLEMS",
    id = "4",
    title = "Median of Two Sorted Arrays",
    topic = "algorithm",
    difficulty = "hard",
    tags = "array, BinarySearch, DivideAndConquer",
    note = "Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.

The overall run time complexity should be O(log (m+n)).

Constraints:
nums1.length == m
nums2.length == n
0 <= m <= 1000
0 <= n <= 1000
1 <= m + n <= 2000
-10^6 <= nums1[i], nums2[i] <= 10^6

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/median-of-two-sorted-arrays
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn find_median_sorted_arrays(nums1: Vec<i32>, nums2: Vec<i32>) -> f64 {
    let (n1, n2) = match (nums1.len(), nums2.len()) {
        (0, 0) => {
            return 0f64;
        }
        (0, 1) => {
            return nums2[0] as f64;
        }
        (1, 0) => {
            return nums1[0] as f64;
        }
        (1, 1) => {
            return (nums1[0] as f64 + nums2[0] as f64) / 2f64;
        }
        _ => (nums1.len(), nums2.len()),
    };

    let (mid, mut cnt) = ((n1 + n2 + 1) >> 1, 0);
    let (mut i1, mut i2) = (nums1.iter().peekable(), nums2.iter().peekable());
    let mut res = Vec::with_capacity(2);

    loop {
        let min = match (i1.peek(), i2.peek()) {
            (Some(&&e1), Some(&&e2)) => {
                if e1 <= e2 {
                    i1.next();
                    e1
                } else {
                    i2.next();
                    e2
                }
            }
            (Some(&&e1), None) => {
                i1.next();
                e1
            }
            (None, Some(&&e2)) => {
                i2.next();
                e2
            }
            (None, None) => {
                break;
            }
        };

        cnt += 1;
        match cnt.cmp(&mid) {
            Ordering::Less => {}
            Ordering::Equal => {
                res.push(min);
                if (n1 + n2) & 1 != 0 {
                    break;
                }
            }
            Ordering::Greater => {
                res.push(min);
                break;
            }
        }
    }

    let len = res.len() as f64;
    res.into_iter().map(|x| x as f64).sum::<f64>() / len
}

#[inject_description(
    problems = "PROBLEMS",
    id = "10",
    title = "Regular Expression Matching",
    topic = "algorithm",
    difficulty = "hard",
    tags = "Recursion,String,DynamicProgramming",
    note = "Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*' where:

'.' Matches any single character.
'*' Matches zero or more of the preceding element.
The matching should cover the entire input string (not partial).

Constraints:
1. 1 <= s.length <= 20
2. 1 <= p.length <= 30
3. s contains only lowercase English letters.
4. p contains only lowercase English letters, '.', and '*'.
5. It is guaranteed for each appearance of the character '*', there will be a previous valid character to match.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/regular-expression-matching
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn is_match(s: String, p: String) -> bool {
    // f[i][j] means that the first 'i' chars of 's' matches the first 'j' chars of 'p'
    // f[i][j] need to satisfy f[i-1][j-1] and s[i] = s[j] when p[j] not equal to '*';
    // if p[j] == '*' and s[i] != p[j-1], f[i][j] need to satisfy f[i][j-2];
    // if p[j] == '*' and s[i] == p[j-1], f[i][j] need to satisfy f[i-1][j] that means '*' matches
    // some characters, or f[i][j] need to satisfy f[i][j-2] that means '*' doesn't matches following chars;
    let (s, p) = match (s.is_empty(), p.is_empty()) {
        (true, true) => {
            return true;
        }
        (false, true) | (true, false) => {
            return false;
        }
        (false, false) => (s.as_bytes(), p.as_bytes()),
    };

    let (m, n) = (s.len(), p.len());
    let mut f = vec![vec![false; n + 1]; m + 1];
    f[0][0] = true;

    let (mut pre_s, mut siter) = (None, s.iter());
    for i in 0..=m {
        let mut pre_p = None;
        for (j, &pc) in p.iter().enumerate() {
            if pc == b'*' {
                f[i][j + 1] |= f[i][j - 1];
                if pre_s.is_some() && (pre_p == Some(b'.') || pre_s == pre_p) {
                    f[i][j + 1] |= f[i - 1][j + 1];
                }
            } else if pre_s.is_some() && (pc == b'.' || pre_s == Some(pc)) {
                f[i][j + 1] |= f[i - 1][j];
            }

            pre_p = Some(pc);
        }

        pre_s = siter.next().copied();
    }

    f[m][n]
}

#[inject_description(
    problems = "PROBLEMS",
    id = "23",
    title = "Merge k Sorted Lists",
    topic = "algorithm",
    difficulty = "hard",
    tags = "",
    note = "You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.

Merge all the linked-lists into one sorted linked-list and return it.

Constraints:

k == lists.length
0 <= k <= 10^4
0 <= lists[i].length <= 500
-10^4 <= lists[i][j] <= 10^4
lists[i] is sorted in ascending order.
The sum of lists[i].length will not exceed 10^4.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/merge-k-sorted-lists
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn merge_k_lists(lists: Vec<Option<Box<ListNode>>>) -> Option<Box<ListNode>> {
    let mut heap = BinaryHeap::new();
    for mut list in lists {
        while let Some(mut node) = list {
            list = node.next.take();
            heap.push(ListNodeMinHeap(node));
        }
    }

    let mut fake_head = Box::new(ListNode::new(0));
    let mut tail = &mut fake_head;
    while let Some(node) = heap.pop() {
        tail.next.replace(node.0);
        tail = tail.next.as_mut().unwrap_or_else(|| unreachable!());
    }

    fake_head.next
}

#[inject_description(
    problems = "PROBLEMS",
    id = "23",
    title = "Merge k Sorted Lists",
    topic = "algorithm",
    difficulty = "hard",
    tags = "",
    note = ""
)]
pub struct ListNodeMinHeap(Box<ListNode>);

impl PartialEq for ListNodeMinHeap {
    fn eq(&self, other: &Self) -> bool {
        self.0.val == other.0.val
    }
}

impl Eq for ListNodeMinHeap {}

impl PartialOrd for ListNodeMinHeap {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "23",
    title = "Merge k Sorted Lists",
    topic = "algorithm",
    difficulty = "hard",
    tags = "",
    note = ""
)]
impl Ord for ListNodeMinHeap {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.0.val.cmp(&other.0.val) {
            Ordering::Less => Ordering::Greater,
            Ordering::Equal => Ordering::Equal,
            Ordering::Greater => Ordering::Less,
        }
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "30",
    title = "Substring with Concatenation of All Words",
    topic = "algorithm",
    difficulty = "hard",
    tags = "HashTable,String,SlidingWindow",
    note = "You are given a string s and an array of strings words of the same length. Return all starting indices of substring(s) in s that is a concatenation of each word in words exactly once, in any order, and without any intervening characters.

You can return the answer in any order.

Constraints:
words[i] consists of lower-case English letters.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/substring-with-concatenation-of-all-words
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn find_substring(s: String, words: Vec<String>) -> Vec<i32> {
    let (s, words) = (
        s.as_bytes(),
        words
            .into_iter()
            .map(|s| s.as_bytes().to_vec())
            .collect::<Vec<_>>(),
    );

    let (word_len, total_len, nums) = match words.last() {
        Some(word) => (word.len(), words.len() * word.len(), words.len()),
        None => {
            return vec![];
        }
    };

    let (mut idx, mut res) = (0, Vec::new());
    let (mut tables, mut buf) = (HashMap::new(), Vec::with_capacity(nums));
    for word in words {
        tables
            .entry(word)
            .and_modify(|e: &mut (i32, i32)| {
                e.0 += 1;
            })
            .or_insert((1, 0));
    }

    while idx + total_len <= s.len() {
        let mut cnt = 0;
        buf.clear();
        for i in 0..nums {
            let sub_s = &s[(idx + (i * word_len)..(idx + (i * word_len) + word_len))];
            match tables.get_mut(sub_s) {
                None => {
                    break;
                }
                Some(entry) => {
                    if entry.0 == entry.1 {
                        break;
                    } else {
                        buf.push(sub_s);
                        entry.1 += 1;
                        cnt += 1;
                    }
                }
            }
        }

        for &e in buf.iter() {
            if let Some(entry) = tables.get_mut(e) {
                entry.1 = 0;
            }
        }

        if cnt == nums {
            res.push(idx as i32);
        }

        idx += 1;
    }

    res
}

#[inject_description(
    problems = "PROBLEMS",
    id = "32",
    title = "Longest Valid Parentheses",
    topic = "algorithm",
    difficulty = "hard",
    tags = "Stack,String,DynamicProgramming",
    note = "Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.

Constraints:
0 <= s.length <= 3 * 10^4
s[i] is '(', or ')'.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/longest-valid-parentheses
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn longest_valid_parentheses(s: String) -> i32 {
    let s = s.as_bytes();
    s.iter()
        .enumerate()
        .fold((vec![-1], 0), |(mut b, mut max), (i, &c)| {
            let i = i as i32;

            if c == b'(' {
                b.push(i);
            } else {
                b.pop();
                match b.last() {
                    None => {
                        b.push(i);
                    }
                    Some(&right_idx) => {
                        max = std::cmp::max(max, i - right_idx);
                    }
                }
            }
            (b, max)
        })
        .1
}

#[inject_description(
    problems = "PROBLEMS",
    id = "37",
    title = "Sudoku Solver",
    topic = "algorithm",
    difficulty = "hard",
    tags = "Array,Backtracking,Matrix",
    note = "Write a program to solve a Sudoku puzzle by filling the empty cells.

A sudoku solution must satisfy all of the following rules:

Each of the digits 1-9 must occur exactly once in each row.
Each of the digits 1-9 must occur exactly once in each column.
Each of the digits 1-9 must occur exactly once in each of the 9 3x3 sub-boxes of the grid.
The '.' character indicates empty cells.

Constraints:
board[i][j] is a digit or '.'.
It is guaranteed that the input board has only one solution.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/sudoku-solver
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn solve_sudoku(board: &mut Vec<Vec<char>>) {
    debug_assert!(board.len() == 9 && board.last().map(|x| x.len()) == Some(9));

    fn dfs(
        space: &[(usize, usize)],
        pos: usize,
        board: &mut Vec<Vec<char>>,
        row: &mut [[bool; 9]; 9],
        column: &mut [[bool; 9]; 9],
        block: &mut [[[bool; 9]; 3]; 3],
        is_exit: &mut bool,
    ) {
        if pos == space.len() {
            *is_exit = true;
        } else {
            let (i, j) = space[pos];
            for digit in 0..9 {
                if *is_exit {
                    break;
                } else if !row[i][digit] && !column[j][digit] && !block[i / 3][j / 3][digit] {
                    (row[i][digit], column[j][digit], block[i / 3][j / 3][digit]) =
                        (true, true, true);
                    board[i][j] = char::from(b'0' + 1u8 + digit as u8);
                    dfs(space, pos + 1, board, row, column, block, is_exit);
                    (row[i][digit], column[j][digit], block[i / 3][j / 3][digit]) =
                        (false, false, false);
                }
            }
        }
    }

    let (mut row, mut col, mut block, mut is_exit) = (
        [[false; 9]; 9],
        [[false; 9]; 9],
        [[[false; 9]; 3]; 3],
        false,
    );

    let mut spaces = Vec::new();
    board.iter().enumerate().for_each(|(i, v)| {
        v.iter().enumerate().for_each(|(j, &c)| {
            if c == '.' {
                spaces.push((i, j));
            } else {
                let digit = (c as u8 - b'0' - 1u8) as usize;
                (row[i][digit], col[j][digit], block[i / 3][j / 3][digit]) = (true, true, true);
            }
        })
    });

    dfs(
        spaces.as_slice(),
        0,
        board,
        &mut row,
        &mut col,
        &mut block,
        &mut is_exit,
    );
}

#[inject_description(
    problems = "PROBLEMS",
    id = "41",
    title = "First Missing Positive",
    topic = "algorithm",
    difficulty = "hard",
    tags = "array, hashtable",
    note = "Given an unsorted integer array nums, return the smallest missing positive integer.

You must implement an algorithm that runs in O(n) time and uses constant extra space.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/first-missing-positive
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn first_missing_positive(mut nums: Vec<i32>) -> i32 {
    // It's easy to prove the `res` in the range `[1,nums.len()+1]`;
    let len = nums.len() as i32;

    nums.iter_mut().for_each(|e| {
        if *e <= 0 {
            *e = len + 1;
        }
    });
    (0..len as usize).for_each(|i| {
        let idx = nums[i].abs() - 1;
        if idx < len {
            nums[idx as usize] = -nums[idx as usize].abs();
        }
    });

    nums.into_iter()
        .enumerate()
        .find(|&(_, ele)| ele > 0)
        .map(|x| x.0 as i32 + 1)
        .unwrap_or_else(|| len + 1)
}

#[inject_description(
    problems = "PROBLEMS",
    id = "42",
    title = "Trapping Rain Water",
    topic = "algorithm",
    difficulty = "hard",
    tags = "Stack, Array, TwoPointers, DynamicProgramming, MonotonicStack",
    note = "Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/trapping-rain-water
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn trap(height: Vec<i32>) -> i32 {
    let (mut res, mut stk) = (0, vec![]);

    for (i, h) in height.into_iter().enumerate() {
        while let Some(&(_, top)) = stk.last() {
            if h > top {
                stk.pop();
                if let Some(&(left, lh)) = stk.last() {
                    res += (i - left - 1) as i32 * (std::cmp::min(lh, h) - top)
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        stk.push((i, h));
    }

    res
}

#[inject_description(
    problems = "PROBLEMS",
    id = "44",
    title = "Wildcard Matching",
    topic = "algorithm",
    difficulty = "hard",
    tags = "Greedy,Recursion,String,DynamicProgramming",
    note = "Given an input string (s) and a pattern (p), implement wildcard pattern matching with support for '?' and '*' where:

'?' Matches any single character.
'*' Matches any sequence of characters (including the empty sequence).
The matching should cover the entire input string (not partial).
Constraints:

0 <= s.length, p.length <= 2000
s contains only lowercase English letters.
p contains only lowercase English letters, '?' or '*'.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/wildcard-matching
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn is_match_ii(s: String, p: String) -> bool {
    // dp[i][j] = s[i] == s[j] && dp[i-1][j-1] when p[j] is a lowercase letters;
    // dp[i][j] = dp[i-1][j-1] when p[j] is the char of '?';
    // dp[i][j] = dp[i][j-1] or dp[i-1][j] when p[j] is the car of '*';
    // dp[i][j] = false, otherwise;
    let (s, p) = (s.as_bytes(), p.as_bytes());
    let mut dp = vec![vec![false; p.len() + 1]; s.len() + 1];
    dp[0][0] = true;
    for (x, &y) in dp[0].iter_mut().skip(1).zip(p.iter()) {
        if y == b'*' {
            *x = true;
        } else {
            break;
        }
    }

    s.iter().enumerate().for_each(|(i, &sc)| {
        p.iter().enumerate().for_each(|(j, &pc)| {
            if pc == b'*' {
                dp[i + 1][j + 1] = dp[i + 1][j] | dp[i][j + 1];
            } else if pc == b'?' || sc == pc {
                dp[i + 1][j + 1] = dp[i][j];
            }
        })
    });

    dp[s.len()][p.len()]
}

#[inject_description(
    problems = "PROBLEMS",
    id = "51",
    title = "N-Queens",
    topic = "algorithm",
    difficulty = "hard",
    tags = "Array,Backtracking",
    note = "The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other.

Given an integer n, return all distinct solutions to the n-queens puzzle. You may return the answer in any order.

Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' both indicate a queen and an empty space, respectively.

Constraints:
1 <= n <= 9

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/n-queens
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn solve_n_queens(n: i32) -> Vec<Vec<String>> {
    fn solve(
        s: &mut Vec<Vec<String>>,
        queens: &mut Vec<Option<usize>>,
        n: usize,
        row: usize,
        col: &mut HashSet<usize>,
        diag1: &mut HashSet<usize>,
        diag2: &mut HashSet<usize>,
    ) {
        if row == n {
            let mut res = Vec::with_capacity(n);
            queens.iter().for_each(|&ele| {
                let mut line = vec![b'.'; n];
                if let Some(idx) = ele {
                    line[idx] = b'Q';
                }
                res.push(unsafe { String::from_utf8_unchecked(line) });
            });
            s.push(res);
        } else {
            for i in 0..n {
                if col.contains(&i)
                    || diag1.contains(&(row.wrapping_sub(i)))
                    || diag2.contains(&(row + i))
                {
                    continue;
                } else {
                    queens[row] = Some(i);
                    col.insert(i);
                    diag1.insert(row.wrapping_sub(i));
                    diag2.insert(row + i);
                    solve(s, queens, n, row + 1, col, diag1, diag2);
                    queens[row] = None;
                    col.remove(&i);
                    diag1.remove(&(row.wrapping_sub(i)));
                    diag2.remove(&(row + i));
                }
            }
        }
    }

    let (mut s, mut queens, n, row, mut cols, mut diag1, mut diag2) = (
        Vec::new(),
        vec![None; n as usize],
        n as usize,
        0,
        HashSet::new(),
        HashSet::new(),
        HashSet::new(),
    );

    solve(
        &mut s,
        &mut queens,
        n,
        row,
        &mut cols,
        &mut diag1,
        &mut diag2,
    );
    s
}

#[inject_description(
    problems = "PROBLEMS",
    id = "52",
    title = "N-Queens II",
    topic = "algorithm",
    difficulty = "hard",
    tags = "Backtracking",
    note = "The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other.

Given an integer n, return the number of distinct solutions to the n-queens puzzle.


Constraints:
1 <= n <= 9

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/n-queens-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn total_n_queens(n: i32) -> i32 {
    fn solve(
        row: i32,
        n: i32,
        col: &mut HashSet<i32>,
        diag1: &mut HashSet<i32>,
        diag2: &mut HashSet<i32>,
    ) -> i32 {
        let mut res = 0;
        if row == n {
            res += 1;
        } else {
            for i in 0..n {
                if col.contains(&i) || diag1.contains(&(row - i)) || diag2.contains(&(row + i)) {
                    continue;
                } else {
                    col.insert(i);
                    diag1.insert(row - i);
                    diag2.insert(row + i);
                    res += solve(row + 1, n, col, diag1, diag2);
                    col.remove(&i);
                    diag1.remove(&(row - i));
                    diag2.remove(&(row + i));
                }
            }
        }
        res
    }

    let (mut col, mut diag1, mut diag2) = (HashSet::new(), HashSet::new(), HashSet::new());
    solve(0, n, &mut col, &mut diag1, &mut diag2)
}

#[inject_description(
    problems = "PROBLEMS",
    id = "60",
    title = "Permutation Sequence",
    topic = "algorithm",
    difficulty = "hard",
    tags = "Recursion, Math",
    note = "The set [1, 2, 3, ..., n] contains a total of n! unique permutations.

By listing and labeling all of the permutations in order, we get the following sequence for n = 3:

\"123\"
\"132\"
\"213\"
\"231\"
\"312\"
\"321\"
Given n and k, return the kth permutation sequence.

Constraints:
1 <= n <= 9
1 <= k <= n!

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/permutation-sequence
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn get_permutation(n: i32, k: i32) -> String {
    if k < 1 {
        String::new()
    } else {
        let mut nums = (1..=n).collect::<Vec<_>>();
        let mut fac = Vec::with_capacity(n as usize);
        fac.push(1);
        (1..n).for_each(|i| fac.push(fac[i as usize - 1] * i));

        let (mut res, mut k) = (String::new(), k - 1);
        (0..n).rev().for_each(|i| {
            let idx = k / fac[i as usize];
            res.push_str(format!("{}", nums.remove(idx as usize)).as_str());
            k -= idx * fac[i as usize];
        });

        res
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "65",
    title = "Valid Number",
    topic = "algorithm",
    difficulty = "hard",
    tags = "String",
    note = "A valid number can be split up into these components (in order):

A decimal number or an integer.
(Optional) An 'e' or 'E', followed by an integer.
A decimal number can be split up into these components (in order):

(Optional) A sign character (either '+' or '-').
One of the following formats:
One or more digits, followed by a dot '.'.
One or more digits, followed by a dot '.', followed by one or more digits.
A dot '.', followed by one or more digits.
An integer can be split up into these components (in order):

(Optional) A sign character (either '+' or '-').
One or more digits.
For example, all the following are valid numbers: [\"2\", \"0089\", \"-0.1\", \"+3.14\", \"4.\", \"-.9\", \"2e10\", \"-90E3\", \"3e+7\", \"+6e-1\", \"53.5e93\", \"-123.456e789\"], while the following are not valid numbers: [\"abc\", \"1a\", \"1e\", \"e3\", \"99e2.5\", \"--6\", \"-+3\", \"95a54e53\"].

Given a string s, return true if s is a valid number.

Constraints:
1 <= s.length <= 20
s consists of only English letters (both uppercase and lowercase), digits (0-9), plus '+', minus '-', or dot '.'.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/valid-number
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn is_number(s: String) -> bool {
    #[derive(Copy, Clone, Eq, PartialEq, Hash)]
    enum Char {
        Number,
        Exp,
        Point,
        Sign,
    }

    #[derive(Copy, Clone, Eq, PartialEq, Hash)]
    enum State {
        Init,
        IntegerSign,
        Integer,
        PointWithoutInt,
        PointWithInt,
        FloatNumber,
        Exp,
        ExpSign,
        ExpNumber,
    }

    // cur_state -> next_state
    let states = vec![
        (
            State::Init,
            vec![
                (Char::Number, State::Integer),
                (Char::Point, State::PointWithoutInt),
                (Char::Sign, State::IntegerSign),
            ],
        ),
        (
            State::IntegerSign,
            vec![
                (Char::Number, State::Integer),
                (Char::Point, State::PointWithoutInt),
            ],
        ),
        (
            State::Integer,
            vec![
                (Char::Number, State::Integer),
                (Char::Point, State::PointWithInt),
                (Char::Exp, State::Exp),
            ],
        ),
        (
            State::PointWithoutInt,
            vec![(Char::Number, State::FloatNumber)],
        ),
        (
            State::PointWithInt,
            vec![(Char::Number, State::FloatNumber), (Char::Exp, State::Exp)],
        ),
        (
            State::FloatNumber,
            vec![(Char::Number, State::FloatNumber), (Char::Exp, State::Exp)],
        ),
        (
            State::Exp,
            vec![
                (Char::Sign, State::ExpSign),
                (Char::Number, State::ExpNumber),
            ],
        ),
        (State::ExpSign, vec![(Char::Number, State::ExpNumber)]),
        (State::ExpNumber, vec![(Char::Number, State::ExpNumber)]),
    ]
    .into_iter()
    .map(|(key, val)| (key, val.into_iter().collect::<HashMap<_, _>>()))
    .collect::<HashMap<_, _>>();

    let mut cur_state = State::Init;
    for c in s.chars() {
        let t = match c {
            x if x.is_ascii_digit() => Char::Number,
            '.' => Char::Point,
            y if y == 'e' || y == 'E' => Char::Exp,
            z if z == '+' || z == '-' => Char::Sign,
            _ => {
                return false;
            }
        };

        if let Some(sub_tables) = states.get(&cur_state) {
            if let Some(&nxt) = sub_tables.get(&t) {
                cur_state = nxt;
            } else {
                return false;
            }
        } else {
            return false;
        }
    }

    cur_state == State::Integer
        || cur_state == State::FloatNumber
        || cur_state == State::PointWithInt
        || cur_state == State::ExpNumber
}

#[inject_description(
    problems = "PROBLEMS",
    id = "68",
    title = "Text Justification",
    topic = "algorithm",
    difficulty = "hard",
    tags = "Array, String, Simulation",
    note = "Given an array of strings words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.

You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces ' ' when necessary so that each line has exactly maxWidth characters.

Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line does not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.

For the last line of text, it should be left-justified, and no extra space is inserted between words.

Note:

A word is defined as a character sequence consisting of non-space characters only.
Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
The input array words contains at least one word.

Constraints:
1 <= words.length <= 300
1 <= words[i].length <= 20
words[i] consists of only English letters and symbols.
1 <= maxWidth <= 100
words[i].length <= maxWidth

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/text-justification
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn full_justify(words: Vec<String>, max_width: i32) -> Vec<String> {
    let (mut res, n, mut right, max_width) = (vec![], words.len(), 0, max_width as usize);
    loop {
        let (left, mut sum_len) = (right, 0);
        for word in words.iter().skip(right) {
            // It's need at least one space between word.
            if sum_len + word.len() + right - left <= max_width {
                right += 1;
                sum_len += word.len();
            } else {
                break;
            }
        }

        if right == n {
            let mut last = words[left..right].join(" ");
            (last.len()..max_width).for_each(|_| last.push(' '));
            res.push(last);
            return res;
        }

        let (num_words, num_space) = (right - left, max_width - sum_len);

        if num_words == 1 {
            let mut line = words[left].clone();
            (0..num_space).for_each(|_| line.push(' '));
            res.push(line);
        } else {
            let (avg_spaces, extra_spaces) =
                (num_space / (num_words - 1), num_space % (num_words - 1));
            let mut sep = String::with_capacity(avg_spaces);
            (0..avg_spaces).for_each(|_| sep.push(' '));
            let s1 = words[(left + extra_spaces + 1)..right].join(sep.as_str());
            sep.push(' ');
            let mut s2 = words[left..(left + extra_spaces + 1)].join(sep.as_str());
            sep.pop();
            s2.push_str(sep.as_str());
            s2.push_str(s1.as_str());
            res.push(s2);
        }
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "72",
    title = "Edit Distance",
    topic = "algorithm",
    difficulty = "hard",
    tags = "String,DynamicProgramming",
    note = "Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.

You have the following three operations permitted on a word:

Insert a character
Delete a character
Replace a character

Constraints:
0 <= word1.length, word2.length <= 500
word1 and word2 consist of lowercase English letters.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/edit-distance
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn min_distance(word1: String, word2: String) -> i32 {
    // dp[i][j] = min(dp[i][j-1]+1, dp[i-1][j]+1, dp[i-1][j-1] + (w1[i] != w2[j]) as i32);
    let (word1, word2) = (word1.as_bytes(), word2.as_bytes());
    let (m, n) = (word1.len(), word2.len());
    if m * n == 0 {
        return m as i32 + n as i32;
    }

    let mut dp = vec![vec![0; n + 1]; m + 1];
    dp.iter_mut()
        .enumerate()
        .for_each(|(i, v)| v.iter_mut().take(1).for_each(|e| *e = i));
    dp.iter_mut()
        .take(1)
        .for_each(|v| v.iter_mut().enumerate().for_each(|(j, e)| *e = j));

    word1.iter().enumerate().for_each(|(i, &w1)| {
        word2.iter().enumerate().for_each(|(j, &w2)| {
            dp[i + 1][j + 1] = std::cmp::min(
                std::cmp::min(dp[i + 1][j] + 1, dp[i][j + 1] + 1),
                dp[i][j] + (w1 != w2) as usize,
            );
        })
    });

    dp[m][n] as i32
}

#[inject_description(
    problems = "PROBLEMS",
    id = "76",
    title = "Minimum Window Substring",
    topic = "algorithm",
    difficulty = "hard",
    tags = "HashTable, string,SlidingWindow",
    note = "Given two strings s and t of lengths m and n respectively, return the minimum window substring of s such that every character in t (including duplicates) is included in the window. If there is no such substring, return the empty string \"\".

The testcases will be generated such that the answer is unique.

A substring is a contiguous sequence of characters within the string.

Constraints:
s and t consist of uppercase and lowercase English letters.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/minimum-window-substring
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn min_window(s: String, t: String) -> String {
    let (ss, ts) = (s.as_bytes(), t.as_bytes());
    let (mut start, mut len) = (0, usize::MAX);
    let (mut left, mut right) = (0, 0);

    let (mut sh, mut th) = (HashMap::new(), HashMap::new());
    ts.iter().for_each(|&c| {
        th.entry(c).and_modify(|x| *x += 1).or_insert(1);
    });

    let check_eq = |a: &HashMap<u8, i32>, b: &HashMap<u8, i32>| {
        if a.len() != b.len() {
            false
        } else {
            for (ka, &va) in a.iter() {
                if b.get(ka).copied().unwrap_or_else(|| unreachable!()) > va {
                    return false;
                }
            }
            true
        }
    };

    while right < ss.len() {
        let mut is_eq = false;
        for (i, c) in ss.iter().enumerate().skip(right) {
            if th.contains_key(c) {
                sh.entry(*c).and_modify(|x| *x += 1).or_insert(1);
            }

            if check_eq(&sh, &th) {
                is_eq = true;
                right = i + 1;
                break;
            }
        }

        if !is_eq {
            break;
        } else if right - left < len {
            start = left;
            len = right - left;
        }

        for c in ss.iter().skip(left).take(right - left) {
            left += 1;
            let (va, vb) = if let Some(&val) = sh.get(c) {
                (val, th.get(c).copied().unwrap_or_else(|| unreachable!()))
            } else {
                if right - left < len {
                    start = left;
                    len = right - left;
                }
                continue;
            };

            if va - vb > 1 {
                if right - left < len {
                    start = left;
                    len = right - left;
                }
                sh.entry(*c).and_modify(|x| *x -= 1);
            } else {
                if right - (left - 1) < len {
                    start = left - 1;
                    len = right - (left - 1);
                }

                if va == 1 {
                    sh.remove(c);
                } else {
                    sh.entry(*c).and_modify(|x| *x -= 1);
                }
                break;
            }
        }
    }

    s[start..(start + if len == usize::MAX { 0 } else { len })].to_string()
}

#[inject_description(
    problems = "PROBLEMS",
    id = "84",
    title = "Largest Rectangle in Histogram",
    topic = "algorithm",
    difficulty = "hard",
    tags = "Stack, Array, MonotonicStack",
    note = "Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.

 

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/largest-rectangle-in-histogram
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn largest_rectangle_area(heights: Vec<i32>) -> i32 {
    let (mut stk, mut left, mut right) = (
        vec![],
        vec![0; heights.len()],
        vec![heights.len() as i32; heights.len()],
    );

    for (i, &h) in heights.iter().enumerate() {
        while let Some(&x) = stk.last() {
            if h <= heights[x] {
                right[x] = i as i32;
                stk.pop();
            } else {
                break;
            }
        }
        left[i] = if let Some(&x) = stk.last() {
            x as i32
        } else {
            -1
        };
        stk.push(i);
    }

    left.into_iter()
        .zip(right.into_iter().zip(heights.into_iter()))
        .fold(0, |res, (l, (r, h))| std::cmp::max(res, h * (r - l - 1)))
}

#[inject_description(
    problems = "PROBLEMS",
    id = "85",
    title = "Maximal Rectangle",
    topic = "algorithm",
    difficulty = "hard",
    tags = "Stack, Array, DynamicProgramming, Matrix, MonotonicStack",
    note = "Given a rows x cols binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.

Constriants:
matrix[i][j] is '0' or '1'.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/maximal-rectangle
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn maximal_rectangle(matrix: Vec<Vec<char>>) -> i32 {
    let (m, n) = if matrix.is_empty() {
        return 0;
    } else {
        (
            matrix.len(),
            matrix
                .last()
                .map(|x| x.len())
                .unwrap_or_else(|| unreachable!()),
        )
    };

    let mut left = vec![vec![0i32; n]; m];

    matrix.iter().enumerate().for_each(|(i, v)| {
        v.iter().enumerate().for_each(|(j, &ele)| {
            if ele == '1' {
                left[i][j] = if j == 0 { 0 } else { left[i][j - 1] } + 1;
            }
        })
    });

    let mut res = 0;
    let (mut stk, mut up, mut down) = (vec![], Vec::with_capacity(m), Vec::with_capacity(m));
    for j in 0..n {
        up.clear();
        down.clear();
        up.resize(m, 0);
        down.resize(m, 0);

        stk.clear();
        for i in 0..m {
            while let Some(&x) = stk.last() {
                let (t1, t2): (&Vec<i32>, &Vec<i32>) = (&left[x], &left[i]);
                if t1[j] >= t2[j] {
                    stk.pop();
                } else {
                    break;
                }
            }
            up[i] = if let Some(&x) = stk.last() {
                x as i32
            } else {
                -1
            };
            stk.push(i);
        }

        stk.clear();
        for i in (0..m).rev() {
            while let Some(&x) = stk.last() {
                if left[x][j] >= left[i][j] {
                    stk.pop();
                } else {
                    break;
                }
            }
            down[i] = if let Some(&x) = stk.last() {
                x as i32
            } else {
                m as i32
            };
            stk.push(i);
        }

        res = up
            .iter()
            .zip(down.iter())
            .enumerate()
            .fold(res, |max, (i, (&u, &d))| {
                std::cmp::max(max, (d - u - 1) * left[i][j])
            });
    }

    res
}

#[inject_description(
    problems = "PROBLEMS",
    id = "87",
    title = "Scramble String",
    topic = "algorithm",
    difficulty = "hard",
    tags = "String, DynamicProgramming",
    note = "We can scramble a string s to get a string t using the following algorithm:

If the length of the string is 1, stop.
If the length of the string is > 1, do the following:
Split the string into two non-empty substrings at a random index, i.e., if the string is s, divide it to x and y where s = x + y.
Randomly decide to swap the two substrings or to keep them in the same order. i.e., after this step, s may become s = x + y or s = y + x.
Apply step 1 recursively on each of the two substrings x and y.
Given two strings s1 and s2 of the same length, return true if s2 is a scrambled string of s1, otherwise, return false.

Constraints:
1 <= s1.length <= 30
s1 and s2 consist of lowercase english letters.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/scramble-string
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn is_scramble(s1: String, s2: String) -> bool {
    let len = if s1.len() == s2.len() {
        s1.len()
    } else {
        return false;
    };

    assert!(s1.len() <= 30 && !s1.is_empty());

    // let mut f = vec![vec![vec![false; len]; len]; len+1];
    let mut f = [[[false; 30]; 30]; 31];
    let (s1, s2): (Vec<char>, Vec<char>) = (s1.chars().collect(), s2.chars().collect());

    f[1].iter_mut().zip(s1.iter()).for_each(|(fi, &x)| {
        fi.iter_mut().zip(s2.iter()).for_each(|(fij, &y)| {
            *fij = x == y;
        })
    });

    (1..=len).for_each(|n| {
        (0..=(len - n)).for_each(|i| {
            (0..=(len - n)).for_each(|j| {
                for k in 1..n {
                    if (f[k][i][j] && f[n - k][i + k][j + k])
                        || (f[k][i][j + n - k] && f[n - k][i + k][j])
                    {
                        f[n][i][j] = true;
                        break;
                    }
                }
            });
        });
    });

    f[len][0][0]
}
