use crate::easy::p1::{ListNode, TreeNode};
use crate::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::{Arc, RwLock};

lazy_static::lazy_static! {
    pub static ref PROBLEMS: Arc<RwLock<Problems<Problem>>> =Arc::new(RwLock::new(Problems::new()));
}

#[inject_description(
    problems = "PROBLEMS",
    id = "228",
    title = "Summary Ranges",
    topic = "algorithm",
    difficulty = "easy",
    tags = "Array",
    note = "You are given a sorted unique integer array nums.

A range [a,b] is the set of all integers from a to b (inclusive).

Return the smallest sorted list of ranges that cover all the numbers in the array exactly. That is, each element of nums is covered by exactly one of the ranges, and there is no integer x such that x is in one of the ranges but not in nums.

Each range [a,b] in the list should be output as:

\"a->b\" if a != b
\"a\" if a == b

Constraints:
All the values of nums are unique.
nums is sorted in ascending order.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/summary-ranges
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn summary_ranges(nums: Vec<i32>) -> Vec<String> {
    let (pre1, pre2) = if let Some(first) = nums.first().copied() {
        (first, first)
    } else {
        return vec![];
    };

    let (mut res, p1, p2) =
        nums.into_iter()
            .fold((vec![], pre1, pre2), |(mut res, pre1, pre2), ele| {
                if pre2 + 1 < ele {
                    let s = if pre1 == pre2 {
                        format!("{}", pre1)
                    } else {
                        format!("{}->{}", pre1, pre2)
                    };
                    res.push(s);
                    (res, ele, ele)
                } else {
                    (res, pre1, ele)
                }
            });

    if p1 == p2 {
        res.push(format!("{}", p1));
    } else {
        res.push(format!("{}->{}", p1, p2));
    }
    res
}

#[inject_description(
    problems = "PROBLEMS",
    id = "231",
    title = "Power of Two",
    topic = "algorithm",
    difficulty = "easy",
    tags = "BitManipulation, Recursion, Math",
    note = "Given an integer n, return true if it is a power of two. Otherwise, return false.

An integer n is a power of two, if there exists an integer x such that n == 2x.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/power-of-two
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn is_power_of_two(n: i32) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

#[inject_description(
    problems = "PROBLEMS",
    id = "234",
    title = "Palindrome Linked List",
    topic = "algorithm",
    difficulty = "easy",
    tags = "Stack, Recursion, LinkedList, TwoPointers",
    note = "Given the head of a singly linked list, return true if it is a palindrome.

Constraints:

The number of nodes in the list is in the range [1, 10^5].
0 <= Node.val <= 9

链接：https://leetcode.cn/problems/palindrome-linked-list
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn is_palindrome(head: Option<Box<ListNode>>) -> bool {
    let (mut buf, mut head) = (vec![], head.as_ref());

    while let Some(node) = head {
        buf.push(node.val);
        head = node.next.as_ref();
    }

    match buf.len() {
        0 | 1 => true,
        l if (l & 1) != 0 => false,
        len => buf
            .iter()
            .take(len >> 1)
            .zip(buf.iter().skip(len >> 1).rev())
            .all(|(&x, &y)| x == y),
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "235",
    title = "Lowest Common Ancestor of a Binary Search Tree",
    topic = "algorithm",
    difficulty = "easy",
    tags = "Tree, DepthFirstSearch, BinarySearchTree, BinaryTree",
    note = "Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

Constraints:
The number of nodes in the tree is in the range [2, 10^5].
-10^9 <= Node.val <= 10^9
All Node.val are unique.
p != q
p and q will exist in the BST.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-search-tree
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn lowest_common_ancestor(
    mut root: Option<Rc<RefCell<TreeNode>>>,
    p: Option<Rc<RefCell<TreeNode>>>,
    q: Option<Rc<RefCell<TreeNode>>>,
) -> Option<Rc<RefCell<TreeNode>>> {
    debug_assert!(root.is_some() && p.is_some() && q.is_some());

    let (p, q) = (
        p.unwrap_or_else(|| unreachable!()).borrow().val,
        q.unwrap_or_else(|| unreachable!()).borrow().val,
    );
    let (p, q) = if p < q { (p, q) } else { (q, p) };
    while let Some(node) = root {
        let v = node.borrow().val;
        if (v > p && v < q) || v == p || v == q {
            return Some(Rc::new(RefCell::new(TreeNode::new(v))));
        } else if q < v {
            root = node.borrow().left.clone();
        } else {
            root = node.borrow().right.clone();
        }
    }

    None
}

#[inject_description(
    problems = "PROBLEMS",
    id = "242",
    title = "Valid Anagram",
    topic = "algorithm",
    difficulty = "easy",
    tags = "HashTable, String, Sorting",
    note = "Given two strings s and t, return true if t is an anagram of s, and false otherwise.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

Constraints:
- s and t consist of lowercase English letters.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/valid-anagram
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn is_anagram(s: String, t: String) -> bool {
    if s.len() != t.len() {
        false
    } else {
        let base = b'a';
        let (mut s_tables, mut t_tables) = ([0; 26], [0; 26]);
        for (&es, &et) in s.as_bytes().iter().zip(t.as_bytes().iter()) {
            s_tables[(es - base) as usize] += 1;
            t_tables[(et - base) as usize] += 1;
        }

        let (mut hs, mut ht) = (
            std::collections::hash_map::DefaultHasher::new(),
            std::collections::hash_map::DefaultHasher::new(),
        );
        s_tables.hash(&mut hs);
        t_tables.hash(&mut ht);
        hs.finish() == ht.finish()
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "257",
    title = "Binary Tree Paths",
    topic = "algorithm",
    difficulty = "easy",
    tags = "Tree, DepthFirstSearch, String, Backtracking, BinaryTree",
    note = "Given the root of a binary tree, return all root-to-leaf paths in any order.

A leaf is a node with no children.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/binary-tree-paths
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn binary_tree_paths(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<String> {
    if let Some(root) = root {
        let mut stk = vec![(root, String::new())];
        let mut res = vec![];

        while let Some((node, mut s)) = stk.pop() {
            s.push_str(node.borrow().val.to_string().as_str());
            let (l, r) = (node.borrow().left.clone(), node.borrow().right.clone());
            if l.is_none() && r.is_none() {
                res.push(s);
            } else {
                s.push_str("->");
                match (l, r) {
                    (Some(n), None) => {
                        stk.push((n, s));
                    }
                    (None, Some(n)) => {
                        stk.push((n, s));
                    }
                    (Some(n1), Some(n2)) => {
                        stk.push((n1, s.clone()));
                        stk.push((n2, s));
                    }
                    (None, None) => {
                        unreachable!()
                    }
                }
            }
        }

        res
    } else {
        vec![]
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "258",
    title = "Add Digits",
    topic = "algorithm",
    difficulty = "easy",
    tags = "Math, NumberTheory, Simulation",
    note = "Given an integer num, repeatedly add all its digits until the result has only one digit, and return it.

Constraints:
0 <= num <= 2^31 - 1
Follow up: Could you do it without any loop/recursion in O(1) runtime?

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/add-digits
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn add_digits(mut num: i32) -> i32 {
    let mut buf = vec![];
    loop {
        if num < 10 {
            return num;
        }

        buf.clear();
        while num > 0 {
            buf.push(num % 10);
            num /= 10;
        }
        num = buf.iter().sum();
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "263",
    title = "Ugly Number",
    topic = "algorithm",
    difficulty = "easy",
    tags = "Math",
    note = "An ugly number is a positive integer whose prime factors are limited to 2, 3, and 5.

Given an integer n, return true if n is an ugly number.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/ugly-number
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn is_ugly(mut n: i32) -> bool {
    if n < 1 {
        return false;
    }

    for d in [2, 3, 5] {
        while n % d == 0 {
            n /= d;
        }
    }

    n == 1
}

#[inject_description(
    problems = "PROBLEMS",
    id = "268",
    title = "Missing Number",
    topic = "algorithm",
    difficulty = "easy",
    tags = "BitManipulation, Array, HashTable, Math, BinarySearch, Sorting",
    note = "Given an array nums containing n distinct numbers in the range [0, n], return the only number in the range that is missing from the array.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/missing-number
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn missing_number(nums: Vec<i32>) -> i32 {
    let s = (((nums.len() + 1) * nums.len()) >> 1) as i32;
    s - nums.into_iter().sum::<i32>()
}

#[allow(non_snake_case)]
fn isBadversion(_n: i32) -> bool {
    unimplemented!("this implemented by the official");
}

#[inject_description(
    problems = "PROBLEMS",
    id = "278",
    title = "First Bad Version",
    topic = "algorithm",
    difficulty = "easy",
    tags = "BinarySearch, Interactive",
    note = "You are a product manager and currently leading a team to develop a new product. Unfortunately, the latest version of your product fails the quality check. Since each version is developed based on the previous version, all the versions after a bad version are also bad.

Suppose you have n versions [1, 2, ..., n] and you want to find out the first bad one, which causes all the following ones to be bad.

You are given an API bool isBadVersion(version) which returns whether version is bad. Implement a function to find the first bad version. You should minimize the number of calls to the API.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/first-bad-version
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn first_bad_version(n: i32) -> i32 {
    let (mut left, mut right) = (1, n);
    while left < right {
        let mid = left + ((right - left) >> 1);
        if isBadversion(mid) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }

    left
}

#[inject_description(
    problems = "PROBLEMS",
    id = "283",
    title = "Move Zeroes",
    topic = "algorithm",
    difficulty = "easy",
    tags = "Array, TwoPointers",
    note = "Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements.

Note that you must do this in-place without making a copy of the array.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/move-zeroes
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn move_zeroes(nums: &mut Vec<i32>) {
    let (mut l, mut r) = match nums.iter().enumerate().find(|&(_, &x)| x == 0) {
        Some((i, _)) => (i, i),
        None => {
            return;
        }
    };

    while (r + 1) < nums.len() {
        match nums.iter().enumerate().skip(r + 1).find(|&(_, &x)| x != 0) {
            Some((i, _)) => {
                nums.swap(l, i);
                l += 1;
                r += 1;
            }
            None => {
                return;
            }
        }
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "290",
    title = "Word Pattern",
    topic = "algorithm",
    difficulty = "easy",
    tags = "HashTable, String",
    note = "Given a pattern and a string s, find if s follows the same pattern.

Here follow means a full match, such that there is a bijection between a letter in pattern and a non-empty word in s.

Constraints:
1 <= pattern.length <= 300
pattern contains only lower-case English letters.
1 <= s.length <= 3000
s contains only lowercase English letters and spaces ' '.
s does not contain any leading or trailing spaces.
All the words in s are separated by a single space.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/word-pattern
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn word_pattern(pattern: String, s: String) -> bool {
    let s = s.split(' ');
    if pattern.len() != s.clone().count() {
        false
    } else {
        let (mut hp, mut hs) = (HashMap::new(), HashMap::new());

        for (i, c) in pattern.chars().enumerate() {
            hp.entry(c)
                .and_modify(|x: &mut Vec<usize>| x.push(i))
                .or_insert_with(|| vec![i]);
        }

        for (i, x) in s.clone().enumerate() {
            hs.entry(x)
                .and_modify(|x: &mut Vec<usize>| x.push(i))
                .or_insert_with(|| vec![i]);
        }

        for (c, x) in pattern.chars().zip(s) {
            if hp.get(&c) != hs.get(x) {
                return false;
            }
        }

        true
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "292",
    title = "Nim Game",
    topic = "algorithm",
    difficulty = "easy",
    tags = "Math, GameTheory, Brainteaser",
    note = "You are playing the following Nim Game with your friend:

Initially, there is a heap of stones on the table.
You and your friend will alternate taking turns, and you go first.
On each turn, the person whose turn it is will remove 1 to 3 stones from the heap.
The one who removes the last stone is the winner.
Given n, the number of stones in the heap, return true if you can win the game assuming both you and your friend play optimally, otherwise return false.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/nim-game
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn can_win_nim(n: i32) -> bool {
    n % 4 != 0
}
