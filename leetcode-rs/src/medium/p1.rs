use crate::easy::p1::ListNode;
use crate::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::ops::Index;
use std::sync::{Arc, RwLock};

lazy_static::lazy_static! {
    pub static ref PROBLEMS: Arc<RwLock<Problems<Problem>>> =Arc::new(RwLock::new(Problems::new()));
}

#[inject_description(
    problems = "PROBLEMS",
    id = "2",
    title = "Add Two Numbers",
    topic = "algorithm",
    difficulty = "medium",
    tags = "Recursion,LinkedList,Math",
    note = "You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/add-two-numbers
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn add_two_numbers(
    mut l1: Option<Box<ListNode>>,
    mut l2: Option<Box<ListNode>>,
) -> Option<Box<ListNode>> {
    let (mut fake_head, mut carry) = (Box::new(ListNode::new(0)), 0);
    let mut tail = &mut fake_head;

    loop {
        let (val, mut node) = match (l1, l2) {
            (Some(mut l), Some(mut r)) => {
                l1 = l.next.take();
                l2 = r.next.take();
                (l.val + r.val + carry, l)
            }
            (Some(mut l), None) => {
                l1 = l.next.take();
                l2 = None;
                (l.val + carry, l)
            }
            (None, Some(mut r)) => {
                l1 = None;
                l2 = r.next.take();
                (r.val + carry, r)
            }
            (None, None) => {
                break;
            }
        };

        carry = if val < 10 {
            node.val = val;
            0
        } else {
            node.val = val - 10;
            1
        };
        tail.next.replace(node);
        tail = tail.next.as_mut().unwrap_or_else(|| unreachable!());
    }

    if carry > 0 {
        tail.next.replace(Box::new(ListNode::new(carry)));
    }

    fake_head.next
}

#[inject_description(
    problems = "PROBLEMS",
    id = "3",
    title = "Longest Substring Without Repeating Characters",
    topic = "algorithm",
    difficulty = "medium",
    tags = "HashTable, String, SlidingWindow",
    note = "Given a string s, find the length of the longest substring without repeating characters.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/longest-substring-without-repeating-characters
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn length_of_longest_substring(s: String) -> i32 {
    let (mut max, mut l) = (0, 0);
    let (mut history, mut pos) = (HashMap::new(), HashMap::new());

    while l < s.len() {
        for (i, ele) in s.chars().enumerate().skip(l) {
            if let Some(&idx) = history.get(&ele) {
                for r in l..=idx {
                    if let Some(x) = pos.get(&r) {
                        history.remove(x);
                        pos.remove(&r);
                    }
                }
                l = idx + 1;
                break;
            } else {
                history.insert(ele, i);
                pos.insert(i, ele);
            }

            if history.len() > max {
                max = history.len();
            }
        }
    }

    max as i32
}

#[inject_description(
    problems = "PROBLEMS",
    id = "5",
    title = "Longest Palindromic Substring",
    topic = "algorithm",
    difficulty = "medium",
    tags = "String,DynamicProgramming",
    note = "Given a string s, return the longest palindromic substring in s.

Constraints:
1 <= s.length <= 1000
s consist of only digits and English letters.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/longest-palindromic-substring
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn longest_palindrome(s: String) -> String {
    // Manacher's algorithm
    // 插入了特殊标记#后, 回文个数必然是奇数的. 以某一位置对称的的半轴长必然是原始回文的长度;
    let mut ps = Vec::with_capacity((s.len() << 1) + 3);
    ps.push('^');
    s.chars().for_each(|x| {
        ps.push('#');
        ps.push(x);
    });
    ps.push('#');
    ps.push('$');

    // cnt记录以i位置对称的长度(不包括自身), c记录上一次最长对称的中心位置, r记录遍历过的最远位置
    let (mut cnt, mut c, mut r) = (Vec::new(), 0usize, 0);
    cnt.resize(ps.len(), 0);
    for i in 1..(ps.len() - 1) {
        // 关于c与i对称的位置(c - (i-c))
        let m = (c << 1).wrapping_sub(i);
        // 跳过已经比较过对称的元素, 因为m关于c和i对称, 如果r大于i, 那么m实在以c为中心的对称的轴上的,
        // 那么m对称的轴和以i为对称的轴必然有重叠, 重叠便是min(r-i,cnt[m])
        // r'----xx-m-xx----c--xx-i-xx----r
        cnt[i] = if r > i {
            std::cmp::min(r - i, cnt[m])
        } else {
            0
        };

        // 以T[i]为中心, 向左右两边查找对称
        while ps[i + 1 + cnt[i]] == ps[i - 1 - cnt[i]] {
            cnt[i] += 1;
        }

        if i + cnt[i] > r {
            c = i;
            r = i + cnt[i];
        }
    }

    match cnt.iter().enumerate().max_by(|&x, &y| x.1.cmp(y.1)) {
        Some((center_idx, &max_len)) => s
            .chars()
            .skip((center_idx - 1 - max_len) >> 1)
            .take(max_len)
            .collect(),
        None => String::new(),
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "6",
    title = "ZigZag Conversion",
    topic = "algorithm",
    difficulty = "medium",
    tags = "String",
    note = "The string \"PAYPALISHIRING\" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)

P   A   H   N
A P L S I I G
Y   I   R
And then read line by line: \"PAHNAPLSIIGYIR\"

Write the code that will take a string and make this conversion given a number of rows:

string convert(string s, int numRows);
 

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/zigzag-conversion
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn convert(s: String, num_rows: i32) -> String {
    let (mut res, num_rows) = if s.len() < 2 {
        return s;
    } else {
        (String::with_capacity(s.len()), num_rows as usize)
    };

    for row in 0..num_rows {
        let (mut idx, mut cnt) = (row, 0);
        while idx < s.len() {
            res.push(s.chars().nth(idx).unwrap());
            if row != 0 && row != (num_rows - 1) && ((num_rows - row - 1) << 1) + idx < s.len() {
                res.push(s.chars().nth(idx + ((num_rows - row - 1) << 1)).unwrap());
            }

            cnt += 1;
            idx = ((num_rows - 1) << 1) * cnt + row;
        }
    }

    res
}

#[inject_description(
    problems = "PROBLEMS",
    id = "7",
    title = "Reverse Integer",
    topic = "algorithm",
    difficulty = "medium",
    tags = "math",
    note = "Given a signed 32-bit integer x, return x with its digits reversed. If reversing x causes the value to go outside the signed 32-bit integer range [-2^31, 2^31 - 1], then return 0.

Assume the environment does not allow you to store 64-bit integers (signed or unsigned).

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/reverse-integer
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn reverse(x: i32) -> i32 {
    let (mut x, mut r) = (x, 0);

    while x != 0 {
        if (r > 214748364 || r < -214748364)
            || ((r == 214748364 || r == -214748364) && (x > 7 || x < -8))
        {
            return 0;
        } else {
            r = r * 10 + x % 10;
            x /= 10;
        }
    }

    r
}

#[inject_description(
    problems = "PROBLEMS",
    id = "8",
    title = "String to Integer (atoi)",
    topic = "algorithm",
    difficulty = "medium",
    tags = "string",
    note = "Implement the myAtoi(string s) function, which converts a string to a 32-bit signed integer (similar to C/C++'s atoi function).

The algorithm for myAtoi(string s) is as follows:

Read in and ignore any leading whitespace.
Check if the next character (if not already at the end of the string) is '-' or '+'. Read this character in if it is either. This determines if the final result is negative or positive respectively. Assume the result is positive if neither is present.
Read in next the characters until the next non-digit character or the end of the input is reached. The rest of the string is ignored.
Convert these digits into an integer (i.e. \"123\" -> 123, \"0032\" -> 32). If no digits were read, then the integer is 0. Change the sign as necessary (from step 2).
If the integer is out of the 32-bit signed integer range [-2^31, 2^31 - 1], then clamp the integer so that it remains in the range. Specifically, integers less than -2^31 should be clamped to -2^31, and integers greater than 2^31 - 1 should be clamped to 2^31 - 1.
Return the integer as the final result.
Note:

Only the space character ' ' is considered a whitespace character.
Do not ignore any characters other than the leading whitespace or the rest of the string after the digits.
 

Constraints:
0 <= s.length <= 200
s consists of English letters (lower-case and upper-case), digits (0-9), ' ', '+', '-', and '.'.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/string-to-integer-atoi
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn my_atoi(s: String) -> i32 {
    let (sign, s) = match s.trim_start().as_bytes().first().copied() {
        Some(b'+') => (1, s.trim_start().as_bytes().index(1..)),
        Some(b'-') => (-1, s.trim_start().as_bytes().index(1..)),
        _ => (1, s.trim_start().as_bytes()),
    };

    let mut n = 0i32;
    for &e in s.iter() {
        if e.is_ascii_digit() {
            match n.checked_mul(10).map(|x| {
                x.checked_add(if sign > 0 {
                    (e - b'0') as i32
                } else {
                    -((e - b'0') as i32)
                })
            }) {
                Some(Some(x)) => {
                    n = x;
                }
                _ => {
                    n = n.saturating_add(n);
                    break;
                }
            }
        }
    }

    n
}

#[inject_description(
    problems = "PROBLEMS",
    id = "11",
    title = "Container With Most Water",
    topic = "algorithm",
    difficulty = "medium",
    tags = "greedy,array,TwoPointers",
    note = "You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.

Notice that you may not slant the container.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/container-with-most-water
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn max_area(height: Vec<i32>) -> i32 {
    let (mut start, mut end, mut res) = (
        height.iter().enumerate().peekable(),
        height.iter().enumerate().rev().peekable(),
        0,
    );

    while let (Some(&(li, &le)), Some(&(ri, &re))) = (start.peek(), end.peek()) {
        if li >= ri {
            break;
        }

        if le <= re {
            start.next();
            res = std::cmp::max(le * (ri as i32 - li as i32), res);
        } else {
            end.next();
            res = std::cmp::max(re * (ri as i32 - li as i32), res);
        }
    }

    res
}

#[inject_description(
    problems = "PROBLEMS",
    id = "12",
    title = "Integer to Roman",
    topic = "algorithm",
    difficulty = "medium",
    tags = "HashTable,Math,String",
    note = "Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
For example, 2 is written as II in Roman numeral, just two one's added together. 12 is written as XII, which is simply X + II. The number 27 is written as XXVII, which is XX + V + II.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

I can be placed before V (5) and X (10) to make 4 and 9. 
X can be placed before L (50) and C (100) to make 40 and 90. 
C can be placed before D (500) and M (1000) to make 400 and 900.
Given an integer, convert it to a roman numeral.

Constraints:
1 <= num <= 3999

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/integer-to-roman
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn int_to_roman(mut num: i32) -> String {
    debug_assert!(num > 0);

    const TABLES: [(i32, &str); 13] = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ];

    let mut roman = String::new();
    for (de, le) in TABLES {
        if num > 0 {
            let cnt = num / de;
            num %= de;
            (0..cnt).for_each(|_| roman.push_str(le));
        } else {
            break;
        }
    }

    roman
}

#[inject_description(
    problems = "PROBLEMS",
    id = "15",
    title = "3Sum",
    topic = "algorithm",
    difficulty = "medium",
    tags = "Array,TwoPointers,Sorting",
    note = "Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.

Constraints:
0 <= nums.length <= 3000
-105 <= nums[i] <= 105

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/3sum
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn three_sum(mut nums: Vec<i32>) -> Vec<Vec<i32>> {
    nums.sort_unstable();
    let zero_idx = nums
        .iter()
        .enumerate()
        .rfind(|&(_, &x)| x == 0)
        .map(|x| x.0)
        .unwrap_or_default();
    let mut res = HashSet::new();

    for (li, &le) in nums.iter().enumerate().take(zero_idx + 1) {
        let t1 = 0 - le;
        for (mi, &me) in nums.iter().enumerate().skip(li + 1) {
            let t2 = t1 - me;
            if t2 >= 0 {
                for &re in nums.iter().skip(std::cmp::max(zero_idx, mi + 1)) {
                    if re >= t2 {
                        if re == t2 {
                            res.insert(vec![le, me, re]);
                        }
                        break;
                    }
                }
            } else {
                for &re in nums.iter().take(zero_idx + 1).skip(mi + 1).rev() {
                    if re <= t2 {
                        if re == t2 {
                            res.insert(vec![le, me, re]);
                        }
                        break;
                    }
                }
            }
        }
    }

    res.into_iter().collect::<Vec<_>>()
}

#[inject_description(
    problems = "PROBLEMS",
    id = "16",
    title = "3Sum Closest",
    topic = "algorithm",
    difficulty = "medium",
    tags = "array, TwoPointers, Sorting",
    note = "Given an integer array nums of length n and an integer target, find three integers in nums such that the sum is closest to target.

Return the sum of the three integers.

You may assume that each input would have exactly one solution.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/3sum-closest
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn three_sum_closest(mut nums: Vec<i32>, target: i32) -> i32 {
    debug_assert!(nums.len() >= 3);

    nums.sort_unstable();
    let (mut sum, mut gap) = (0, i32::MAX);
    for (li, &le) in nums.iter().enumerate() {
        for (mi, &me) in nums.iter().enumerate().skip(li + 1) {
            for &re in nums.iter().skip(mi + 1) {
                let tmp = le + me + re;
                let min = target - tmp;
                if min.abs() < gap.abs() {
                    gap = min;
                    sum = tmp;
                } else {
                    break;
                }
            }
        }
    }

    sum
}

#[inject_description(
    problems = "PROBLEMS",
    id = "17",
    title = "Letter Combinations of a Phone Number",
    topic = "algorithm",
    difficulty = "medium",
    tags = "HashTable, String, Backtracking",
    note = "Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

Constraints:
0 <= digits.length <= 4
digits[i] is a digit in the range ['2', '9'].

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/letter-combinations-of-a-phone-number
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn letter_combinations(digits: String) -> Vec<String> {
    const TABLES: [&str; 8] = ["abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"];

    let (codes, mut res) = match digits.is_empty() {
        true => {
            return vec![];
        }
        false => {
            let codes = digits
                .as_bytes()
                .iter()
                .map(|&x| TABLES[(x - b'2') as usize])
                .collect::<Vec<_>>();
            let elems = codes.iter().fold(1, |b, x| b * x.len());
            (codes, vec![String::new(); elems])
        }
    };

    codes.into_iter().for_each(|s| {
        res.chunks_mut(s.len())
            .for_each(|r| r.iter_mut().zip(s.chars()).for_each(|(r, c)| r.push(c)))
    });

    res
}

#[inject_description(
    problems = "PROBLEMS",
    id = "18",
    title = "4Sum",
    topic = "algorithm",
    difficulty = "medium",
    tags = "Array,TwoPointers,Sorting",
    note = "Given an array nums of n integers, return an array of all the unique quadruplets [nums[a], nums[b], nums[c], nums[d]] such that:

0 <= a, b, c, d < n
a, b, c, and d are distinct.
nums[a] + nums[b] + nums[c] + nums[d] == target
You may return the answer in any order.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/4sum
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn four_sum(mut nums: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
    nums.sort_unstable();
    let tgt_idx = nums
        .iter()
        .enumerate()
        .rfind(|&(_, &x)| x == target)
        .map(|x| x.0)
        .unwrap_or_else(|| nums.len().saturating_sub(1));
    let zero_idx = nums
        .iter()
        .enumerate()
        .find(|&(_, &x)| x == 0)
        .map(|x| x.0)
        .unwrap_or_else(|| nums.len().saturating_sub(1));
    let mut res = HashSet::new();

    for (i1, &e1) in nums.iter().enumerate().take(tgt_idx + 1) {
        let t1 = target - e1;
        for (i2, &e2) in nums.iter().enumerate().skip(i1 + 1) {
            let t2 = t1 - e2;
            for (i3, &e3) in nums.iter().enumerate().skip(i2 + 1) {
                let t3 = t2 - e3;
                if t3 >= 0 {
                    for &e4 in nums.iter().skip(std::cmp::max(zero_idx, i3 + 1)) {
                        if e4 >= t3 {
                            if e4 == t3 {
                                res.insert(vec![e1, e2, e3, e4]);
                            }
                            break;
                        }
                    }
                } else {
                    for &e4 in nums.iter().take(zero_idx).skip(i3 + 1) {
                        if e4 <= t3 {
                            if e4 == t3 {
                                res.insert(vec![e1, e2, e3, e4]);
                            }
                            break;
                        }
                    }
                }
            }
        }
    }

    res.into_iter().collect()
}

#[inject_description(
    problems = "PROBLEMS",
    id = "19",
    title = "Remove Nth Node From End of List",
    topic = "algorithm",
    difficulty = "medium",
    tags = "LinkedList, TwoPointers",
    note = "Given the head of a linked list, remove the nth node from the end of the list and return its head.
Constraints:
The number of nodes in the list is sz.
1 <= sz <= 30
0 <= Node.val <= 100
1 <= n <= sz

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/remove-nth-node-from-end-of-list
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn remove_nth_from_end(head: Option<Box<ListNode>>, n: i32) -> Option<Box<ListNode>> {
    let (mut list, mut cnt) = (head.as_ref(), 0);
    while let Some(node) = list {
        cnt += 1;
        list = node.next.as_ref();
    }

    let (n, mut res) = (cnt - n, Box::new(ListNode::new(0)));
    res.next = head;
    let mut pre = &mut res;

    for _ in 0..n {
        pre = pre.next.as_mut().unwrap_or_else(|| unreachable!());
    }
    let node = pre
        .next
        .as_mut()
        .unwrap_or_else(|| unreachable!())
        .next
        .take();
    pre.next = node;

    res.next
}

#[inject_description(
    problems = "PROBLEMS",
    id = "22",
    title = "Generate Parentheses",
    topic = "algorithm",
    difficulty = "medium",
    tags = "String, DynamicProgramming, Backtracking",
    note = "Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/generate-parentheses
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn generate_parenthesis(n: i32) -> Vec<String> {
    debug_assert!(n > 0);
    // F[n] = F[x] + F[y], x + y = n - 1;
    let n = n as usize;
    let mut f = vec![vec![String::new()]];

    for i in 1..=n {
        // (F[x].zip(F[y])
        let mut f_n = Vec::new();
        f.iter()
            .take(i)
            .zip(f.iter().take(i).rev())
            .for_each(|(f_x, f_y)| {
                f_x.iter()
                    .for_each(|x| f_y.iter().for_each(|y| f_n.push(format!("({}){}", x, y))));
            });

        f.push(f_n)
    }

    f.pop().unwrap_or_default()
}

#[inject_description(
    problems = "PROBLEMS",
    id = "24",
    title = "Swap Nodes in Pairs",
    topic = "algorithm",
    difficulty = "medium",
    tags = "Recursion,LinkedList",
    note = "Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)

 

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/swap-nodes-in-pairs
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn swap_pairs(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut fake_head = Box::new(ListNode::new(0));
    fake_head.next = head;
    let mut pre = &mut fake_head;
    loop {
        // pre->cur->next->tail..
        if pre.next.as_ref().map(|cur| cur.next.is_some()) == Some(true) {
            let mut cur = pre.next.take().unwrap_or_else(|| unreachable!());
            let mut next = cur.next.take().unwrap_or_else(|| unreachable!());
            cur.next = next.next.take();
            next.next = Some(cur);
            pre.next = Some(next);
            pre = pre
                .next
                .as_mut()
                .map(|cur| cur.next.as_mut().unwrap_or_else(|| unreachable!()))
                .unwrap_or_else(|| unreachable!());
        } else {
            break;
        }
    }

    fake_head.next
}

#[inject_description(
    problems = "PROBLEMS",
    id = "29",
    title = "Divide Two Integers",
    topic = "algorithm",
    difficulty = "medium",
    tags = "Math, BitManipulation",
    note = "Given two integers dividend and divisor, divide two integers without using multiplication, division, and mod operator.

The integer division should truncate toward zero, which means losing its fractional part. For example, 8.345 would be truncated to 8, and -2.7335 would be truncated to -2.

Return the quotient after dividing dividend by divisor.

Note: Assume we are dealing with an environment that could only store integers within the 32-bit signed integer range: [−2^31, 2^31 − 1]. For this problem, if the quotient is strictly greater than 2^31 - 1, then return 2^31 - 1, and if the quotient is strictly less than -2^31, then return -2^31.

Constraints:
divisor != 0

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/divide-two-integers
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn divide(dividend: i32, divisor: i32) -> i32 {
    debug_assert_ne!(divisor, 0);

    let is_neg = dividend.is_positive() ^ divisor.is_positive();
    let (dividend, divisor) = (
        if dividend == i32::MIN {
            i32::MAX as u32 + 1
        } else {
            dividend.abs() as u32
        },
        if divisor == i32::MIN {
            i32::MAX as u32 + 1
        } else {
            divisor.abs() as u32
        },
    );

    if dividend == 0 || divisor > dividend {
        0
    } else if divisor == dividend {
        if is_neg {
            -1
        } else {
            1
        }
    } else {
        let (mut sum, mut res, mut bit_len) = (
            0,
            0,
            (u32::BITS - dividend.leading_zeros()) - (u32::BITS - divisor.leading_zeros()),
        );

        // y = a*x, x = 2^{?} + 2^{?} + ...
        loop {
            if bit_len == 0 {
                if sum + divisor <= dividend {
                    res += 1
                };

                return if is_neg {
                    if res >= (i32::MAX as u32 + 1) {
                        i32::MIN
                    } else {
                        -(res as i32)
                    }
                } else if res >= (i32::MAX as u32) {
                    i32::MAX
                } else {
                    res as i32
                };
            }

            let tmp = (divisor << bit_len) + sum;
            if tmp <= dividend {
                res += 1 << bit_len;
                sum = tmp;
            }

            bit_len -= 1;
        }
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "31",
    title = "Next Permutation",
    topic = "algorithm",
    difficulty = "medium",
    tags = "Array, TwoPointers",
    note = "A permutation of an array of integers is an arrangement of its members into a sequence or linear order.

For example, for arr = [1,2,3], the following are considered permutations of arr: [1,2,3], [1,3,2], [3,1,2], [2,3,1].
The next permutation of an array of integers is the next lexicographically greater permutation of its integer. More formally, if all the permutations of the array are sorted in one container according to their lexicographical order, then the next permutation of that array is the permutation that follows it in the sorted container. If such arrangement is not possible, the array must be rearranged as the lowest possible order (i.e., sorted in ascending order).

For example, the next permutation of arr = [1,2,3] is [1,3,2].
Similarly, the next permutation of arr = [2,3,1] is [3,1,2].
While the next permutation of arr = [3,2,1] is [1,2,3] because [3,2,1] does not have a lexicographical larger rearrangement.
Given an array of integers nums, find the next permutation of nums.

The replacement must be in place and use only constant extra memory.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/next-permutation
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn next_permutation(nums: &mut Vec<i32>) {
    // find a[i] < a[i+1] from tail to head
    let idx_a = nums
        .iter()
        .enumerate()
        .rev()
        .skip(1)
        .zip(nums.iter().rev())
        .find(|((_, a), b)| a < b)
        .map(|((i_a, &a), _)| (i_a, a));

    // find a[i] < a[j] in the range of [i+1, n) from tail to head
    if let Some((i_a, a)) = idx_a {
        if let Some(i_b) = nums
            .iter()
            .enumerate()
            .skip(i_a + 1)
            .rev()
            .find(|(_, &b)| a < b)
            .map(|(i_b, _)| i_b)
        {
            nums.swap(i_a, i_b);
        }

        // a[i+1] is descending, so need to convert an ascending sub-seq
        let s = &mut nums.as_mut_slice()[(i_a + 1)..];
        s.reverse();
    } else {
        nums.reverse();
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "33",
    title = "Search in Rotated Sorted Array",
    topic = "algorithm",
    difficulty = "medium",
    tags = "array, binarySearch",
    note = "There is an integer array nums sorted in ascending order (with distinct values).

Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].

Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

You must write an algorithm with O(log n) runtime complexity.

Constraints:
1 <= nums.length <= 5000
-10^4 <= nums[i] <= 10^4
All values of nums are unique.
nums is an ascending array that is possibly rotated.
-10^4 <= target <= 10^4

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/search-in-rotated-sorted-array
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn search(nums: Vec<i32>, target: i32) -> i32 {
    if nums.len() < 2 {
        if nums.first().copied() == Some(target) {
            0
        } else {
            -1
        }
    } else {
        let (mut l, mut r) = (0, nums.len() - 1);
        while l <= r {
            let mid = (l + r) >> 1;
            if nums[mid] == target {
                return mid as i32;
            } else if nums[0] <= nums[mid] {
                if nums[0] <= target && target < nums[mid] {
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            } else if nums[mid] < target && target <= nums[nums.len() - 1] {
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        -1
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "34",
    title = "Find First and Last Position of Element in Sorted Array",
    topic = "algorithm",
    difficulty = "medium",
    tags = "Array, BinarySearch",
    note = "Given an array of integers nums sorted in non-decreasing order, find the starting and ending position of a given target value.

If target is not found in the array, return [-1, -1].

You must write an algorithm with O(log n) runtime complexity.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn search_range(nums: Vec<i32>, target: i32) -> Vec<i32> {
    let (mut left, mut right, mut idx) = (0i32, nums.len() as i32 - 1, None);

    // find the target
    while left <= right {
        let mid = (left + right) >> 1;
        match nums[mid as usize].cmp(&target) {
            Ordering::Less => {
                left = mid + 1;
            }
            Ordering::Equal => {
                idx = Some(mid as usize);
                break;
            }
            Ordering::Greater => {
                right = mid - 1;
            }
        }
    }

    if let Some(i) = idx {
        match (
            nums.iter()
                .enumerate()
                .take(i + 1)
                .rev()
                .find(|&(_, &e)| e != target)
                .map(|e| e.0),
            nums.iter()
                .enumerate()
                .skip(i + 1)
                .find(|&(_, &e)| e != target)
                .map(|e| e.0),
        ) {
            (Some(x), Some(y)) => {
                vec![x as i32 + 1, y as i32 - 1]
            }
            (Some(x), None) => {
                vec![x as i32 + 1, i as i32]
            }
            (None, Some(y)) => {
                vec![i as i32, y as i32 - 1]
            }
            (None, None) => {
                vec![i as i32, i as i32]
            }
        }
    } else {
        vec![-1, -1]
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "36",
    title = "Valid Sudoku",
    topic = "algorithm",
    difficulty = "medium",
    tags = "Array,HashTable,Matrix",
    note = "Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:

Each row must contain the digits 1-9 without repetition.
Each column must contain the digits 1-9 without repetition.
Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.
Note:

A Sudoku board (partially filled) could be valid but is not necessarily solvable.
Only the filled cells need to be validated according to the mentioned rules.

Constraints:
board.length == 9
board[i].length == 9
board[i][j] is a digit 1-9 or '.'.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/valid-sudoku
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn is_valid_sudoku(board: Vec<Vec<char>>) -> bool {
    const fn cvt(c: char) -> usize {
        match c {
            '1' => 0,
            '2' => 1,
            '3' => 2,
            '4' => 3,
            '5' => 4,
            '6' => 5,
            '7' => 6,
            '8' => 7,
            '9' => 8,
            _ => unsafe {
                std::hint::unreachable_unchecked();
            },
        }
    }

    let (mut rows, mut cols, mut subs) = ([[0; 9]; 9], [[0; 9]; 9], [[[0; 9]; 3]; 3]);
    for (i, v) in board.iter().take(9).enumerate() {
        for (j, &c) in v.iter().take(9).enumerate() {
            if c != '.' {
                let idx = cvt(c);
                rows[i][idx] += 1;
                cols[j][idx] += 1;
                subs[i / 3][j / 3][idx] += 1;

                if rows[i][idx] > 1 || cols[j][idx] > 1 || subs[i / 3][j / 3][idx] > 1 {
                    return false;
                }
            }
        }
    }

    true
}
