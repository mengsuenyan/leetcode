use crate::easy::p1::ListNode;
use crate::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

lazy_static::lazy_static! {
    pub static ref PROBLEMS: Arc<RwLock<Problems<Problem>>> =Arc::new(RwLock::new(Problems::new()));
}

#[inject_description(
    problems = "PROBLEMS",
    id = "38",
    title = "Count and Say",
    topic = "algorithm",
    difficulty = "medium",
    tags = "string",
    note = "The count-and-say sequence is a sequence of digit strings defined by the recursive formula:

countAndSay(1) = \"1\"
countAndSay(n) is the way you would \"say\" the digit string from countAndSay(n-1), which is then converted into a different digit string.
To determine how you \"say\" a digit string, split it into the minimal number of substrings such that each substring contains exactly one unique digit. Then for each substring, say the number of digits, then say the digit. Finally, concatenate every said digit.

For example, the saying and conversion for digit string \"3322251\":

Given a positive integer n, return the nth term of the count-and-say sequence.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/count-and-say
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn count_and_say(n: i32) -> String {
    debug_assert!(n > 0);

    (1..n)
        .fold(("1".to_string(), String::new()), |(pre, mut buf), _| {
            let (mut first, mut cnt) = (pre.chars().next().unwrap(), 1usize);
            buf.clear();
            for c in pre.chars().skip(1) {
                if c == first {
                    cnt += 1;
                } else {
                    buf.push_str(cnt.to_string().as_str());
                    buf.push(first);
                    cnt = 1;
                    first = c;
                }
            }

            buf.push_str(cnt.to_string().as_str());
            buf.push(first);

            (buf, pre)
        })
        .0
}

#[inject_description(
    problems = "PROBLEMS",
    id = "39",
    title = "Combination Sum",
    topic = "algorithm",
    difficulty = "medium",
    tags = "array,backtracking",
    note = "Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.

The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different.

It is guaranteed that the number of unique combinations that sum up to target is less than 150 combinations for the given input.

Constraints:
1 <= candidates.length <= 30
1 <= candidates[i] <= 200
All elements of candidates are distinct.
1 <= target <= 500

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/combination-sum
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn combination_sum(mut candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
    fn dfs(nums: &[i32], tgt: i32, stk: &mut Vec<i32>) -> Vec<Vec<i32>> {
        let mut res = vec![];
        if tgt == 0 {
            res.push(stk.clone());
        } else {
            for (i, &e) in nums.iter().enumerate() {
                if tgt < e {
                    break;
                }
                stk.push(e);
                let mut tmp = dfs(&nums[i..], tgt - e, stk);
                res.append(&mut tmp);
                stk.pop();
            }
        }

        res
    }

    candidates.sort_unstable();
    let mut stk = Vec::new();
    dfs(candidates.as_slice(), target, &mut stk)
}

#[inject_description(
    problems = "PROBLEMS",
    id = "40",
    title = "Combination Sum II",
    topic = "algorithm",
    difficulty = "medium",
    tags = "array, backtracking",
    note = "Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.

Each number in candidates may only be used once in the combination.

Note: The solution set must not contain duplicate combinations.

Constraints:
1 <= candidates.length <= 100
1 <= candidates[i] <= 50
1 <= target <= 30

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/combination-sum-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn combination_sum2(mut candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
    fn dfs(nums: &[i32], tgt: i32, stk: &mut Vec<i32>) -> Vec<Vec<i32>> {
        let mut res = vec![];
        if tgt == 0 {
            res.push(stk.clone());
        } else {
            let mut pre = None;
            for (i, &e) in nums.iter().enumerate() {
                if pre == Some(e) {
                    continue;
                } else if tgt < e {
                    break;
                }

                pre = Some(e);

                stk.push(e);
                let mut tmp = dfs(&nums[(i + 1)..], tgt - e, stk);
                res.append(&mut tmp);
                stk.pop();
            }
        }

        res
    }

    candidates.sort_unstable();
    let mut stk = Vec::new();
    dfs(candidates.as_slice(), target, &mut stk)
}

#[inject_description(
    problems = "PROBLEMS",
    id = "43",
    title = "Multiply Strings",
    topic = "algorithm",
    difficulty = "medium",
    tags = "math, string, simulation",
    note = "Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string.

Note: You must not use any built-in BigInteger library or convert the inputs to integer directly.

Constraints:
1 <= num1.length, num2.length <= 200
num1 and num2 consist of digits only.
Both num1 and num2 do not contain any leading zero, except the number 0 itself.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/multiply-strings
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn multiply(num1: String, num2: String) -> String {
    let mut m = vec![0; num1.len() + num2.len()];
    num1.as_bytes()
        .iter()
        .map(|&c| (c - b'0') as u32)
        .rev()
        .enumerate()
        .for_each(|(i, n1)| {
            num2.as_bytes()
                .iter()
                .map(|&c| (c - b'0') as u32)
                .rev()
                .enumerate()
                .for_each(|(j, n2)| {
                    m[i + j] += n1 * n2;
                    m[i + j + 1] += m[i + j] / 10;
                    m[i + j] %= 10;
                });
        });

    m.into_iter()
        .enumerate()
        .rev()
        .skip_while(|&(i, e)| i != 0 && e == 0)
        .map(|(_, e)| char::from(e as u8 + b'0'))
        .collect()
}

#[inject_description(
    problems = "PROBLEMS",
    id = "45",
    title = "Jump Game II",
    topic = "algorithm",
    difficulty = "medium",
    tags = "Greedy, Array, DynamicProgramming",
    note = "Given an array of non-negative integers nums, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Your goal is to reach the last index in the minimum number of jumps.

You can assume that you can always reach the last index.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/jump-game-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn jump(nums: Vec<i32>) -> i32 {
    debug_assert!(!nums.is_empty());

    let len = nums.len().saturating_sub(1);
    nums.into_iter()
        .enumerate()
        .take(len)
        .fold((0i32, 0i32, 0i32), |(max, end, step), (i, e)| {
            let i = i as i32;
            let new_max = std::cmp::max(max, i + e);
            if i == end {
                (new_max, new_max, step + 1)
            } else {
                (new_max, end, step)
            }
        })
        .2
}

#[inject_description(
    problems = "PROBLEMS",
    id = "46",
    title = "Permutations",
    topic = "algorithm",
    difficulty = "medium",
    tags = "array, backtracking",
    note = "Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.

 

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/permutations
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn permute(nums: Vec<i32>) -> Vec<Vec<i32>> {
    let len = match nums.is_empty() {
        true => {
            return vec![];
        }
        false => (1..=nums.len()).product::<usize>(),
    };

    let mut res = nums
        .iter()
        .cycle()
        .take(len)
        .map(|&x| {
            let mut v = Vec::with_capacity(nums.len());
            v.push(x);
            v
        })
        .collect::<Vec<_>>();

    res.chunks_mut(nums.len())
        .enumerate()
        .for_each(|(i, chunk)| {
            (1..nums.len()).for_each(|j| {
                chunk
                    .iter_mut()
                    .zip(nums.iter().cycle().skip(j * (i + 1)))
                    .for_each(|(c, &e)| c.push(e))
            })
        });

    res
}

#[inject_description(
    problems = "PROBLEMS",
    id = "47",
    title = "Permutations II",
    topic = "algorithm",
    difficulty = "medium",
    tags = "array, backtracking",
    note = "Given a collection of numbers, nums, that might contain duplicates, return all possible unique permutations in any order.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/permutations-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn permute_unique(nums: Vec<i32>) -> Vec<Vec<i32>> {
    let len = match nums.is_empty() {
        true => {
            return vec![];
        }
        false => (1..=nums.len()).product::<usize>(),
    };

    let mut res = nums
        .iter()
        .cycle()
        .take(len)
        .map(|&x| {
            let mut v = Vec::with_capacity(nums.len());
            v.push(x);
            v
        })
        .collect::<Vec<_>>();

    res.chunks_mut(nums.len())
        .enumerate()
        .for_each(|(i, chunk)| {
            (1..nums.len()).for_each(|j| {
                chunk
                    .iter_mut()
                    .zip(nums.iter().cycle().skip(j * (i + 1)))
                    .for_each(|(c, &e)| c.push(e))
            })
        });

    res.sort();
    res.dedup();
    res
}

#[inject_description(
    problems = "PROBLEMS",
    id = "48",
    title = "Rotate Image",
    topic = "algorithm",
    difficulty = "medium",
    tags = "array,math,matrix",
    note = "You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).

You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/rotate-image
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn rotate(matrix: &mut Vec<Vec<i32>>) {
    let n = matrix.len();

    // flip from the top to bottom
    (0..(n >> 1)).for_each(|i| matrix.swap(i, n - 1 - i));

    // flip by the main diagonal
    unsafe {
        (0..n).for_each(|i| {
            let up = matrix.as_mut_ptr().add(i);
            ((i + 1)..n).for_each(|j| {
                let down = matrix.as_mut_ptr().add(j);
                std::ptr::swap((*up).as_mut_ptr().add(j), (*down).as_mut_ptr().add(i));
            })
        });
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "49",
    title = "Group Anagrams",
    topic = "algorithm",
    difficulty = "medium",
    tags = "array, hashtable, string, sorting",
    note = "Given an array of strings strs, group the anagrams together. You can return the answer in any order.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

strs[i] consists of lowercase English letters.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/group-anagrams
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn group_anagrams(strs: Vec<String>) -> Vec<Vec<String>> {
    let mut tables: HashMap<u64, Vec<String>> = HashMap::new();
    for s in strs.into_iter() {
        let mut tmp = [0u8; 26];
        s.as_bytes().iter().for_each(|&c| {
            tmp[(c - b'a') as usize] += 1;
        });
        let mut hasher = DefaultHasher::new();
        tmp.hash(&mut hasher);
        tables.entry(hasher.finish()).or_default().push(s);
    }

    let len = tables.len();
    tables
        .into_iter()
        .fold(Vec::with_capacity(len), |mut b, (_, e)| {
            b.push(e);
            b
        })
}

#[inject_description(
    problems = "PROBLEMS",
    id = "50",
    title = "Pow(x, n)",
    topic = "algorithm",
    difficulty = "medium",
    tags = "",
    note = "Implement pow(x, n), which calculates x raised to the power n (i.e., xn).\
Constraints:

-100.0 < x < 100.0
-2^31 <= n <= 2^31 - 1
-10^4 <= x^n <= 10^4

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/powx-n
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn my_pow(x: f64, n: i32) -> f64 {
    let (mut p, mut res, mut tmp) = ((n as i64).abs(), 1f64, x);

    while p > 0 {
        if (p & 1) == 1 {
            res *= tmp;
        }

        tmp *= tmp;
        p >>= 1;
    }

    if n >= 0 {
        res
    } else {
        1f64 / res
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "54",
    title = "Spiral Matrix",
    topic = "algorithm",
    difficulty = "medium",
    tags = "array, matrix, simulation",
    note = "Given an m x n matrix, return all elements of the matrix in spiral order.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/spiral-matrix
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn spiral_order(matrix: Vec<Vec<i32>>) -> Vec<i32> {
    let (rows, cols) = match matrix.len() {
        0 => {
            return vec![];
        }
        1 => {
            return matrix[0].clone();
        }
        _ if matrix.last().map(|x| x.is_empty()) == Some(true) => {
            return vec![];
        }
        _ => (matrix.len(), matrix.last().unwrap().len()),
    };

    let (mut left, mut right, mut top, mut bottom, mut res) =
        (0, cols - 1, 0, rows - 1, Vec::with_capacity(cols * rows));

    while left <= right && top <= bottom {
        // top
        (left..=right).for_each(|col| res.push(matrix[top][col]));

        // right
        ((top + 1)..=bottom).for_each(|row| res.push(matrix[row][right]));

        // bottom
        ((left + 1)..right)
            .rev()
            .for_each(|col| res.push(matrix[bottom][col]));

        // left
        ((top + 1)..=bottom)
            .rev()
            .for_each(|row| res.push(matrix[row][left]));

        left += 1;
        right -= 1;
        top += 1;
        bottom -= 1;
    }

    res
}

#[inject_description(
    problems = "PROBLEMS",
    id = "55",
    title = "Jump Game",
    topic = "algorithm",
    difficulty = "medium",
    tags = "array, greedy, dynamicprogramming",
    note = "You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position.

Return true if you can reach the last index, or false otherwise.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/jump-game
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn can_jump(nums: Vec<i32>) -> bool {
    nums.iter()
        .enumerate()
        .fold(0, |b, (i, &e)| std::cmp::max(b, i as i32 + e))
        >= nums.len().saturating_sub(1) as i32
}

#[inject_description(
    problems = "PROBLEMS",
    id = "56",
    title = "Merge Intervals",
    topic = "algorithm",
    difficulty = "medium",
    tags = "array, sorting",
    note = "Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/merge-intervals
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn merge(mut intervals: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    intervals.sort_unstable_by_key(|e| e.first().copied());
    intervals
        .into_iter()
        .fold(Vec::new(), |mut res: Vec<Vec<i32>>, range| {
            let (first, last) = (range[0], range[1]);
            if let Some(pre) = res.last_mut() {
                if pre[1] < first {
                    res.push(vec![first, last]);
                } else {
                    pre[1] = std::cmp::max(pre[1], last);
                }
            } else {
                res.push(vec![first, last]);
            }
            res
        })
}

#[inject_description(
    problems = "PROBLEMS",
    id = "57",
    title = "Insert Interval",
    topic = "algorithm",
    difficulty = "medium",
    tags = "array",
    note = "You are given an array of non-overlapping intervals intervals where intervals[i] = [starti, endi] represent the start and the end of the ith interval and intervals is sorted in ascending order by starti. You are also given an interval newInterval = [start, end] that represents the start and end of another interval.

Insert newInterval into intervals such that intervals is still sorted in ascending order by starti and intervals still does not have any overlapping intervals (merge overlapping intervals if necessary).

Return intervals after the insertion.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/insert-interval
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn insert(intervals: Vec<Vec<i32>>, new_interval: Vec<i32>) -> Vec<Vec<i32>> {
    let (mut l, mut r) = (new_interval[0], new_interval[1]);

    let (mut is_insert, mut res) = (false, Vec::with_capacity(intervals.len() + 1));

    intervals.into_iter().for_each(|e| {
        let (first, last) = (e[0], e[1]);
        if first > r {
            if !is_insert {
                is_insert = true;
                res.push(vec![l, r]);
            }
            res.push(e);
        } else if last < l {
            res.push(e);
        } else {
            l = std::cmp::min(l, first);
            r = std::cmp::max(r, last);
        }
    });

    if !is_insert {
        res.push(vec![l, r]);
    }

    res
}

#[inject_description(
    problems = "PROBLEMS",
    id = "59",
    title = "Spiral Matrix II",
    topic = "algorithm",
    difficulty = "medium",
    tags = "array, matrix, simulation",
    note = "Given a positive integer n, generate an n x n matrix filled with elements from 1 to n2 in spiral order.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/spiral-matrix-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn generate_matrix(n: i32) -> Vec<Vec<i32>> {
    let (n, mut cnt) = if n < 0 {
        return vec![];
    } else if n == 1 {
        return vec![vec![1]];
    } else {
        (n as usize, 1)
    };

    let (mut left, mut right, mut top, mut bottom, mut res) =
        (0, n - 1, 0, n - 1, vec![vec![0; n]; n]);
    while left <= right && top <= bottom {
        (left..=right).for_each(|col| {
            res[top][col] = cnt;
            cnt += 1;
        });

        ((top + 1)..=bottom).for_each(|row| {
            res[row][right] = cnt;
            cnt += 1;
        });

        ((left + 1)..right).rev().for_each(|col| {
            res[bottom][col] = cnt;
            cnt += 1;
        });

        ((top + 1)..=bottom).rev().for_each(|row| {
            res[row][left] = cnt;
            cnt += 1;
        });

        left += 1;
        right -= 1;
        top += 1;
        bottom -= 1;
    }

    res
}

#[inject_description(
    problems = "PROBLEMS",
    id = "61",
    title = "Rotate List",
    topic = "algorithm",
    difficulty = "medium",
    tags = "LinkedList, TwoPointers",
    note = "Given the head of a linked list, rotate the list to the right by k places.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/rotate-list
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn rotate_right(mut head: Option<Box<ListNode>>, k: i32) -> Option<Box<ListNode>> {
    debug_assert!(k > -1);
    let (mut nodes, k) = (vec![], k as usize);

    while let Some(mut node) = head {
        head = node.next.take();
        nodes.push(node);
    }

    let k = k % nodes.len();
    nodes.rotate_right(k);
    let mut fake_head = Box::new(ListNode::new(0));
    let mut tail = &mut fake_head;
    for node in nodes {
        tail.next = Some(node);
        tail = tail.next.as_mut().unwrap_or_else(|| unreachable!());
    }

    fake_head.next
}

#[inject_description(
    problems = "PROBLEMS",
    id = "64",
    title = "Unique Paths",
    topic = "algorithm",
    difficulty = "medium",
    tags = "math, DynamicProgramming, Combinatorics",
    note = "There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any point in time.

Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.

The test cases are generated so that the answer will be less than or equal to 2 * 109.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/unique-paths
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn unique_paths(m: i32, n: i32) -> i32 {
    let (m, n) = (m as usize, n as usize);
    let mut history = vec![vec![0; n]; m];

    // f[i,j] = f[i-1,j] + f[i,j-1]
    (0..m).for_each(|i| {
        history[i][0] = 1;
    });
    (0..n).for_each(|j| {
        history[0][j] = 1;
    });
    (1..m).for_each(|i| {
        (1..n).for_each(|j| {
            history[i][j] = history[i - 1][j] + history[i][j - 1];
        })
    });

    history[m - 1][n - 1]
}

#[inject_description(
    problems = "PROBLEMS",
    id = "63",
    title = "Unique Paths II",
    topic = "algorithm",
    difficulty = "medium",
    tags = "array, DynamicProgramming, Matrix",
    note = "You are given an m x n integer array grid. There is a robot initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m-1][n-1]). The robot can only move either down or right at any point in time.

An obstacle and space are marked as 1 or 0 respectively in grid. A path that the robot takes cannot include any square that is an obstacle.

Return the number of possible unique paths that the robot can take to reach the bottom-right corner.

The testcases are generated so that the answer will be less than or equal to 2 * 109.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/unique-paths-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn unique_paths_with_obstacles(obstacle_grid: Vec<Vec<i32>>) -> i32 {
    let m = if obstacle_grid.is_empty() || obstacle_grid.last().map(|c| c.is_empty()) == Some(true)
    {
        return 0;
    } else {
        obstacle_grid.len()
    };

    let mut history = vec![0; m];
    history[0] = obstacle_grid[0][0] ^ 1;
    obstacle_grid.into_iter().for_each(|rows| {
        let mut pre = None;
        rows.into_iter().enumerate().for_each(|(j, is_pass)| {
            if is_pass == 1 {
                history[j] = 0;
            } else if pre == Some(0) {
                history[j] += history[j - 1];
            }
            pre = Some(is_pass)
        });
    });

    history[m - 1]
}

#[inject_description(
    problems = "PROBLEMS",
    id = "64",
    title = "Minimum Path Sum",
    topic = "algorithm",
    difficulty = "medium",
    tags = "Array, DynamicProgramming, Matrix",
    note = "Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/minimum-path-sum
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn min_path_sum(grid: Vec<Vec<i32>>) -> i32 {
    let (rows, cols) = if grid.is_empty() || grid.last().map(|r| r.is_empty()) == Some(true) {
        return 0;
    } else {
        (grid.len(), grid.last().unwrap().len())
    };

    let mut history = vec![vec![0; cols]; rows];
    history.iter_mut().zip(grid.iter()).fold(0, |b, (h, g)| {
        *h.first_mut().unwrap() = g.first().unwrap() + b;
        h.first().copied().unwrap()
    });

    history[0]
        .iter_mut()
        .zip(grid[0].iter())
        .fold(0, |b, (h, &g)| {
            *h = g + b;
            *h
        });

    grid.into_iter().enumerate().skip(1).for_each(|(i, row)| {
        row.into_iter().enumerate().skip(1).for_each(|(j, e)| {
            history[i][j] = std::cmp::min(history[i - 1][j], history[i][j - 1]) + e;
        });
    });

    history[rows - 1][cols - 1]
}

#[inject_description(
    problems = "PROBLEMS",
    id = "71",
    title = "Simplify Path",
    topic = "algorithm",
    difficulty = "medium",
    tags = "Stack, String",
    note = "Given a string path, which is an absolute path (starting with a slash '/') to a file or directory in a Unix-style file system, convert it to the simplified canonical path.

In a Unix-style file system, a period '.' refers to the current directory, a double period '..' refers to the directory up a level, and any multiple consecutive slashes (i.e. '//') are treated as a single slash '/'. For this problem, any other format of periods such as '...' are treated as file/directory names.

The canonical path should have the following format:

The path starts with a single slash '/'.
Any two directories are separated by a single slash '/'.
The path does not end with a trailing '/'.
The path only contains the directories on the path from the root directory to the target file or directory (i.e., no period '.' or double period '..')
Return the simplified canonical path.

Constraints:
1 <= path.length <= 3000
path consists of English letters, digits, period '.', slash '/' or '_'.
path is a valid absolute Unix path.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/simplify-path
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn simplify_path(path: String) -> String {
    // split by the '/'
    let names = path
        .split('/')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    let mut stk = Vec::new();
    for s in names {
        if s == ".." {
            if !stk.is_empty() {
                stk.pop();
            }
        } else if s != "." {
            stk.push(s);
        }
    }

    if stk.is_empty() {
        "/".to_string()
    } else {
        stk.into_iter().fold(String::new(), |mut b, s| {
            b.push('/');
            b.push_str(s);
            b
        })
    }
}
