use crate::prelude::*;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::sync::{Arc, RwLock};

lazy_static::lazy_static! {
    pub static ref PROBLEMS: Arc<RwLock<Problems<Problem>>> =Arc::new(RwLock::new(Problems::new()));
}

#[inject_description(
    problems = "PROBLEMS",
    id = "1",
    title = "Two Sum",
    difficulty = "Easy",
    topic = "Algorithm",
    tags = "Array, HashTable",
    note = "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/two-sum
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
    "
)]
pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
    let mut res = Vec::with_capacity(2);
    let mut h = HashMap::new();

    for (cur_idx, ele) in nums.into_iter().enumerate() {
        match h.get(&ele) {
            Some(&pre_idx) => {
                res.push(pre_idx as i32);
                res.push(cur_idx as i32);
                break;
            }
            None => {
                h.insert(target - ele, cur_idx);
            }
        }
    }

    res
}

#[inject_description(
    problems = "PROBLEMS",
    id = "9",
    title = "Palindrome Number",
    difficulty = "Easy",
    topic = "Algorithm",
    tags = "Math",
    note = "Given an integer x, return true if x is palindrome integer.

An integer is a palindrome when it reads the same backward as forward.

For example, 121 is a palindrome while 123 is not.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/palindrome-number
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn is_palindrome(x: i32) -> bool {
    let x = x.to_string();
    let len = x.len() >> 1;
    for (left, right) in x.chars().take(len).zip(x.chars().rev().take(len)) {
        if left != right {
            return false;
        }
    }

    true
}

#[inject_description(
    problems = "PROBLEMS",
    id = "13",
    title = "Roman to Integer",
    difficulty = "easy",
    topic = "algorithm",
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
Given a roman numeral, convert it to an integer.



来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/roman-to-integer
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn roman_to_int(s: String) -> i32 {
    const fn tables(c: char) -> i32 {
        match c {
            'I' => 1,
            'V' => 5,
            'X' => 10,
            'L' => 50,
            'C' => 100,
            'D' => 500,
            'M' => 1000,
            _ => unreachable!(),
        }
    }

    let first = s.chars().take(1).fold(0, |_a, b| tables(b));
    s.chars()
        .take(s.len().saturating_sub(1))
        .zip(s.chars().skip(1))
        .fold(first, |sum, (cur, next)| {
            sum + if tables(cur) < tables(next) {
                // because the `cur` was added to the `sum`, so need to subtract `cur` for twice.
                tables(next) - (tables(cur) << 1)
            } else {
                tables(next)
            }
        })
}

#[inject_description(
    problems = "PROBLEMS",
    id = "14",
    title = "Longest Common Prefix",
    topic = "algorithm",
    difficulty = "easy",
    tags = "string",
    note = "Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string \"\".

Constraints:

1 <= strs.length <= 200
0 <= strs[i].length <= 200
strs[i] consists of only lower-case English letters.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/longest-common-prefix
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn longest_common_prefix(mut strs: Vec<String>) -> String {
    match strs.pop() {
        None => Default::default(),
        Some(mut first) => {
            if strs.is_empty() {
                return first;
            }

            let mut len = 0;
            let mut fiter = first.as_bytes().iter();
            let mut strs = strs.iter().map(|s| s.as_bytes().iter()).collect::<Vec<_>>();
            'outloop: while fiter.len() != 0 {
                let cur = fiter.next();
                for ele in strs.iter_mut() {
                    if cur != ele.next() {
                        break 'outloop;
                    }
                }

                len += 1;
            }

            first.truncate(len);
            first
        }
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "20",
    title = "Valid Parentheses",
    topic = "algorithm",
    difficulty = "easy",
    tags = "stack, string",
    note = "Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.

Constraints:

1 <= s.length <= 104
s consists of parentheses only '()[]{}'.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/valid-parentheses
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn is_valid(s: String) -> bool {
    let mut stack = Vec::new();
    for c in s.chars() {
        match c {
            '(' | '{' | '[' => {
                stack.push(c);
            }
            ')' => {
                if stack.pop() != Some('(') {
                    return false;
                }
            }
            '}' => {
                if stack.pop() != Some('{') {
                    return false;
                }
            }
            ']' => {
                if stack.pop() != Some('[') {
                    return false;
                }
            }
            _ => {
                unreachable!("s only contain char of bracket");
            }
        }
    }

    stack.is_empty()
}

// Definition for singly-linked list.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    #[inline]
    pub fn new(val: i32) -> Self {
        ListNode { next: None, val }
    }

    pub fn from_slice(v: &[i32]) -> Option<Box<ListNode>> {
        let mut head = Box::new(ListNode::new(0));
        let mut tail = &mut head;
        for &e in v {
            tail.next = Some(Box::new(ListNode::new(e)));
            tail = tail.next.as_mut().unwrap_or_else(|| unreachable!());
        }
        head.next.take()
    }

    pub fn to_pretty_string(&self) -> String {
        let (mut list, mut s) = (self, String::new());
        loop {
            s.push_str(format!("{}->", list.val).as_str());
            match list.next.as_deref() {
                None => {
                    break;
                }
                Some(l) => {
                    list = l;
                }
            }
        }

        if !s.is_empty() {
            s.pop();
            s.pop();
        }

        s
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "20",
    title = "Merge Two Sorted Lists",
    topic = "algorithm",
    difficulty = "easy",
    tags = "recursion, LinkedList",
    note = "You are given the heads of two sorted linked lists list1 and list2.

Merge the two lists in a one sorted list. The list should be made by splicing together the nodes of the first two lists.

Return the head of the merged linked list.
Constraints:

The number of nodes in both lists is in the range [0, 50].
-100 <= Node.val <= 100
Both list1 and list2 are sorted in non-decreasing order.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/merge-two-sorted-lists
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn merge_two_lists(
    mut list1: Option<Box<ListNode>>,
    mut list2: Option<Box<ListNode>>,
) -> Option<Box<ListNode>> {
    let mut head = Box::new(ListNode::new(0));
    let mut tail = &mut head;
    'outloop: loop {
        match list1.as_ref().zip(list2.as_ref()) {
            None => {
                if list1.is_some() {
                    tail.next = list1;
                    tail = tail.next.as_mut().unwrap_or_else(|| unreachable!());
                    list1 = tail.next.take();
                } else if list2.is_some() {
                    tail.next = list2;
                    tail = tail.next.as_mut().unwrap_or_else(|| unreachable!());
                    list2 = tail.next.take();
                } else {
                    break 'outloop;
                }
            }
            Some((l, r)) => {
                if l.val <= r.val {
                    tail.next = list1;
                    tail = tail.next.as_mut().unwrap_or_else(|| unreachable!());
                    list1 = tail.next.take();
                } else {
                    tail.next = list2;
                    tail = tail.next.as_mut().unwrap_or_else(|| unreachable!());
                    list2 = tail.next.take();
                }
            }
        }
    }

    head.next.take()
}

#[inject_description(
    problems = "PROBLEMS",
    id = "26",
    title = "Remove Duplicates from Sorted Array",
    topic = "algorithm",
    difficulty = "easy",
    tags = "array,TwoPointers",
    note = "Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place such that each unique element appears only once. The relative order of the elements should be kept the same.

Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.

Return k after placing the final result in the first k slots of nums.

Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/remove-duplicates-from-sorted-array
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn remove_duplicates(nums: &mut Vec<i32>) -> i32 {
    let (mut new_len, mut read) = (nums.len(), 1);

    if new_len > 1 {
        unsafe {
            let (mut cur, mut next) = (nums.as_mut_ptr(), nums.as_mut_ptr().add(1));
            while read < new_len {
                if cur.read() == next.read() {
                    next.copy_from_nonoverlapping(next.offset(1), new_len - 2);
                    new_len -= 1;
                }

                cur = cur.offset(1);
                next = next.offset(1);
                read += 1;
            }
        }
    }

    nums.truncate(new_len);

    new_len as i32
}

#[inject_description(
    problems = "PROBLEMS",
    id = "27",
    title = "Remove Element",
    topic = "algorithm",
    difficulty = "easy",
    tags = "array, TwoPointers",
    note = "Given an integer array nums and an integer val, remove all occurrences of val in nums in-place. The relative order of the elements may be changed.

Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.

Return k after placing the final result in the first k slots of nums.

Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.

Constraints:

0 <= nums.length <= 100
0 <= nums[i] <= 50
0 <= val <= 100

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/remove-element
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn remove_element(nums: &mut Vec<i32>, val: i32) -> i32 {
    let (mut new_len, mut read) = (nums.len(), 0);

    unsafe {
        let first = nums.as_mut_ptr();
        let mut cur = nums.as_mut_ptr();
        while read < new_len {
            if cur.read() == val {
                cur.swap(first.add(new_len - 1));
                new_len -= 1;
            } else {
                cur = cur.offset(1);
                read += 1;
            }
        }
    }

    nums.truncate(new_len);
    new_len as i32
}

#[inject_description(
    problems = "PROBLEMS",
    id = "28",
    title = "Implement strStr()",
    topic = "algorithm",
    difficulty = "easy",
    tags = "TwoPointers, String, StringMatching",
    note = "Implement strStr().

Given two strings needle and haystack, return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.

Clarification:

What should we return when needle is an empty string? This is a great question to ask during an interview.

For the purpose of this problem, we will return 0 when needle is an empty string. This is consistent to C's strstr() and Java's indexOf().

Constraints:
1 <= haystack.length, needle.length <= 10^4
haystack and needle consist of only lowercase English characters.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/implement-strstr
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn str_str(haystack: String, needle: String) -> i32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    if needle.is_empty() {
        0
    } else if haystack.len() < needle.len() {
        -1
    } else {
        let mut hasher = DefaultHasher::new();
        needle.hash(&mut hasher);
        let target = hasher.finish();

        let hasher = DefaultHasher::new();
        for i in 0..(haystack.len() - needle.len()) {
            let mut hasher = hasher.clone();
            let s = &haystack[i..(i + needle.len())];
            s.hash(&mut hasher);
            if hasher.finish() == target && s == needle {
                return i as i32;
            }
        }

        -1
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "35",
    title = "Search Insert Position",
    topic = "algorithm",
    difficulty = "easy",
    tags = "Array, BinarySearch",
    note = "Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

You must write an algorithm with O(log n) runtime complexity.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/search-insert-position
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn search_insert(nums: Vec<i32>, target: i32) -> i32 {
    for (i, &e) in nums.iter().enumerate() {
        if e >= target {
            return i as i32;
        }
    }

    nums.len() as i32
}

#[inject_description(
    problems = "PROBLEMS",
    id = "53",
    title = "Maximum Subarray",
    topic = "algorithm",
    difficulty = "easy",
    tags = "Array,DivideAndConquer, DynamicProgramming",
    note = "Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

A subarray is a contiguous part of an array.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/maximum-subarray
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn max_sub_array(nums: Vec<i32>) -> i32 {
    // F(n) = max(F(n-1)+A[n], A[n])
    let f = *nums
        .first()
        .unwrap_or_else(|| unreachable!("nums need to contain at least one number"));
    nums.into_iter()
        .skip(1)
        .fold((f, f), |(pre_max, f), e| {
            let f = std::cmp::max(f + e, e);
            (std::cmp::max(f, pre_max), f)
        })
        .0
}

#[inject_description(
    problems = "PROBLEMS",
    id = "58",
    title = "Length of Last Word",
    topic = "algorithm",
    difficulty = "easy",
    tags = "string",
    note = "Given a string s consisting of words and spaces, return the length of the last word in the string.

A word is a maximal substring consisting of non-space characters only.

Constraints:
1 <= s.length <= 10^4
s consists of only English letters and spaces ' '.
There will be at least one word in s.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/length-of-last-word
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn length_of_last_word(s: String) -> i32 {
    let mut len = 0;
    for ele in s.trim().chars().rev() {
        if ele == ' ' {
            break;
        } else {
            len += 1;
        }
    }

    len
}

#[inject_description(
    problems = "PROBLEMS",
    id = "66",
    title = "Plus One",
    topic = "algorithm",
    difficulty = "easy",
    tags = "array, math",
    note = "You are given a large integer represented as an integer array digits, where each digits[i] is the ith digit of the integer. The digits are ordered from most significant to least significant in left-to-right order. The large integer does not contain any leading 0's.

Increment the large integer by one and return the resulting array of digits.

Constraints:
1 <= digits.length <= 100
0 <= digits[i] <= 9
digits does not contain any leading 0's.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/plus-one
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn plus_one(mut digits: Vec<i32>) -> Vec<i32> {
    let mut carry = 1;
    for ele in digits.iter_mut().rev() {
        *ele += carry;
        if *ele > 9 {
            *ele = 0;
            carry = 1;
        } else {
            carry = 0;
        }
    }

    if carry > 0 {
        digits.insert(0, carry);
    }

    digits
}

#[inject_description(
    problems = "PROBLEMS",
    id = "67",
    title = "Add Binary",
    topic = "algorithm",
    difficulty = "easy",
    tags = "BitManipulation, Math, string, Simulation",
    note = "Given two binary strings a and b, return their sum as a binary string.

Constraints:
1 <= a.length, b.length <= 10^4
a and b consist only of '0' or '1' characters.
Each string does not contain leading zeros except for the zero itself.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/add-binary
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
 "
)]
pub fn add_binary(a: String, b: String) -> String {
    let (mut long, short) = if a.len() > b.len() { (a, b) } else { (b, a) };
    let mut is_carry = false;
    unsafe {
        for (l, &s) in long
            .as_bytes_mut()
            .iter_mut()
            .rev()
            .zip(short.as_bytes().iter().rev())
        {
            let (cur, carry) = match (*l, s) {
                (b'0', b'0') => (if is_carry { b'1' } else { b'0' }, false),
                (b'0', b'1') | (b'1', b'0') => {
                    if is_carry {
                        (b'0', true)
                    } else {
                        (b'1', false)
                    }
                }
                (b'1', b'1') => {
                    if is_carry {
                        (b'1', true)
                    } else {
                        (b'0', true)
                    }
                }
                _ => unreachable!(),
            };
            *l = cur;
            is_carry = carry;
        }

        long.as_bytes_mut()
            .iter_mut()
            .rev()
            .skip(short.len())
            .for_each(|l| {
                *l = match *l {
                    b'0' => {
                        if is_carry {
                            is_carry = false;
                            b'1'
                        } else {
                            b'0'
                        }
                    }
                    b'1' => {
                        if is_carry {
                            is_carry = true;
                            b'0'
                        } else {
                            b'1'
                        }
                    }
                    _ => unreachable!(),
                }
            });
    }

    if is_carry {
        long.insert(0, '1');
    }

    long
}

#[inject_description(
    problems = "PROBLEMS",
    id = "69",
    title = "Sqrt(x)",
    topic = "algorithm",
    difficulty = "easy",
    tags = "math, BinarySearch",
    note = "Given a non-negative integer x, compute and return the square root of x.

Since the return type is an integer, the decimal digits are truncated, and only the integer part of the result is returned.

Note: You are not allowed to use any built-in exponent function or operator, such as pow(x, 0.5) or x ** 0.5.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/sqrtx
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn my_sqrt(x: i32) -> i32 {
    let x = x as u64;
    let (mut left, mut right) = (0u64, (x >> 1) + 1);

    while (right - left) > 1 {
        let m = (left + right) >> 1;
        let m2 = m * m;
        match m2.cmp(&x) {
            Ordering::Less => {left = m;}
            Ordering::Equal => {return m as i32;}
            Ordering::Greater => {right = m;}
        }
    }

    let (l2, r2) = (left * left, right * right);
    if l2 <= x && r2 > x {
        left as i32
    } else {
        right as i32
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "70",
    title = "Climbing Stairs",
    topic = "algorithm",
    difficulty = "easy",
    tags = "Memoization, math, DynamicProgramming",
    note = "You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

Constraints:
1 <= n <= 45

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/climbing-stairs
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn climb_stairs(n: i32) -> i32 {
    // c[i] = c[i-1] + c[i-2]
    (1..=n).fold((0, 1), |(prev, cur), _| (cur, cur + prev)).1
}

#[inject_description(
    problems = "PROBLEMS",
    id = "83",
    title = "Remove Duplicates from Sorted List",
    topic = "algorithm",
    difficulty = "easy",
    tags = "LinkedList",
    note = "Given the head of a sorted linked list, delete all duplicates such that each element appears only once. Return the linked list sorted as well.

Constraints:
The number of nodes in the list is in the range [0, 300].
-100 <= Node.val <= 100
The list is guaranteed to be sorted in ascending order.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/remove-duplicates-from-sorted-list
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn delete_duplicates(mut head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut node = head.as_mut();
    while let Some(cur) = node {
        if Some(cur.val) == cur.next.as_ref().map(|x| x.val) {
            let x = cur
                .next
                .as_mut()
                .map(|x| x.next.take())
                .unwrap_or_else(|| unreachable!());
            cur.next = x;
            node = Some(cur);
        } else {
            node = cur.next.as_mut();
        }
    }

    head
}

#[inject_description(
    problems = "PROBLEMS",
    id = "88",
    title = "Merge Sorted Array",
    topic = "algorithm",
    difficulty = "easy",
    tags = "array, TwoPointers, Sorting",
    note = "You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, representing the number of elements in nums1 and nums2 respectively.

Merge nums1 and nums2 into a single array sorted in non-decreasing order.

The final sorted array should not be returned by the function, but instead be stored inside the array nums1. To accommodate this, nums1 has a length of m + n, where the first m elements denote the elements that should be merged, and the last n elements are set to 0 and should be ignored. nums2 has a length of n.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/merge-sorted-array
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
#[allow(clippy::ptr_arg)]
pub fn merge(nums1: &mut Vec<i32>, m: i32, nums2: &mut Vec<i32>, n: i32) {
    let (mut m, mut n) = (m as usize, n as usize);
    nums1.resize((m + n) as usize, Default::default());

    let mut pos = m + n;
    while m > 0 && n > 0 {
        pos -= 1;
        if nums1[m - 1] > nums2[n - 1] {
            nums1[pos] = nums1[m - 1];
            m -= 1;
        } else {
            nums1[pos] = nums2[n - 1];
            n -= 1;
        }
    }

    while m > 0 {
        pos -= 1;
        nums1[pos] = nums1[m - 1];
        m -= 1;
    }

    while n > 0 {
        pos -= 1;
        nums1[pos] = nums2[n - 1];
        n -= 1;
    }
}

// Definition for a binary tree node.
#[derive(Debug, PartialEq, Eq)]
pub struct TreeNode {
    pub val: i32,
    pub left: Option<Rc<RefCell<TreeNode>>>,
    pub right: Option<Rc<RefCell<TreeNode>>>,
}

impl TreeNode {
    #[inline]
    pub fn new(val: i32) -> Self {
        TreeNode {
            val,
            left: None,
            right: None,
        }
    }

    pub fn from_slice(mut v: &[i32]) -> Option<Rc<RefCell<TreeNode>>> {
        if v.is_empty() {
            return None;
        }

        let mut stk = VecDeque::new();
        let root = Rc::new(RefCell::new(TreeNode::new(*v.first().unwrap())));
        stk.push_back(root.clone());
        v = &v[1..];

        while let Some(node) = stk.pop_front() {
            let (l, r) = (v.get(0), v.get(1));
            match l {
                None => {break;}
                Some(&val) => {
                    // `i32::MAX` represent `null` Node, just for test conveniently
                    if val != i32::MAX {
                        let left = Rc::new(RefCell::new(TreeNode::new(val)));
                        node.borrow_mut().left = Some(left.clone());
                        stk.push_back(left);
                    }
                    v = &v[1..];
                }
            }

            match r {
                None => {break;}
                Some(&val) => {
                    if val != i32::MAX {
                        let right = Rc::new(RefCell::new(TreeNode::new(val)));
                        node.borrow_mut().right = Some(right.clone());
                        stk.push_back(right);
                    }
                    v = &v[1..];
                }
            }
        }

        Some(root)
    }

    /// Just for test
    /// The sequence is from top to bottom and left to right. If the leaf is `null` that will fill `i32::MAX`
    pub fn to_vec(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        let (mut res, mut stk) = (Vec::new(), VecDeque::new());
        stk.push_back(root);

        while let Some(ele) = stk.pop_front() {
            match ele {
                None => {
                    // just for test conveniently
                    res.push(i32::MAX);
                }
                Some(node) => {
                    res.push(node.borrow().val);
                    stk.push_back(node.borrow().left.clone());
                    stk.push_back(node.borrow().right.clone());
                }
            }
        }

        // discard all `null` node in the tail.

        while let Some(&ele) = res.last() {
            if ele == i32::MAX {
                res.pop();
            } else {
                break;
            }
        }

        res
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "94",
    title = "Binary Tree Inorder Traversal",
    topic = "algorithm",
    difficulty = "easy",
    tags = "Stack, Tree, DepthFirstSearch, BinaryTree",
    note = "Given the root of a binary tree, return the inorder traversal of its nodes' values.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/binary-tree-inorder-traversal
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn inorder_traversal(mut root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    let (mut stk, mut res) = (Vec::new(), Vec::new());

    while !stk.is_empty() || root.is_some() {
        match root {
            None => {
                root = stk.pop();
                res.push(root.as_ref().unwrap().borrow().val);
                root = root.clone().as_ref().unwrap().borrow().right.clone();
            }
            Some(parent) => {
                root = parent.borrow().left.clone();
                stk.push(parent);
            }
        }
    }

    res
}

#[inject_description(
    problems = "PROBLEMS",
    id = "100",
    title = "Same Tree",
    topic = "algorithm",
    difficulty = "easy",
    tags = "Tree, DepthFirstSearch, BreadthFirstSearch, BinaryTree",
    note = "Given the roots of two binary trees p and q, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/same-tree
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn is_same_tree(p: Option<Rc<RefCell<TreeNode>>>, q: Option<Rc<RefCell<TreeNode>>>) -> bool {
    if p.is_none() && q.is_none() {
        true
    } else {
        p.as_ref().map(|x| x.borrow().val) == q.as_ref().map(|x| x.borrow().val)
            && is_same_tree(
                p.as_ref()
                    .map(|x| x.borrow().left.clone())
                    .unwrap_or_default(),
                q.as_ref()
                    .map(|x| x.borrow().left.clone())
                    .unwrap_or_default(),
            )
            && is_same_tree(
                p.as_ref()
                    .map(|x| x.borrow().right.clone())
                    .unwrap_or_default(),
                q.as_ref()
                    .map(|x| x.borrow().right.clone())
                    .unwrap_or_default(),
            )
    }
}
