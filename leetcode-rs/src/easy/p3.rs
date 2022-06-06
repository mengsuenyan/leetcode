use crate::easy::p1::{ListNode, TreeNode};
use crate::prelude::*;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::{Arc, RwLock};

lazy_static::lazy_static! {
    pub static ref PROBLEMS: Arc<RwLock<Problems<Problem>>> =Arc::new(RwLock::new(Problems::new()));
}

#[inject_description(
    problems = "PROBLEMS",
    id = "171",
    title = "Excel Sheet Column Number",
    topic = "algorithm",
    difficulty = "easy",
    tags = "math, string",
    note = "Given a string columnTitle that represents the column title as appears in an Excel sheet, return its corresponding column number.

For example:

A -> 1
B -> 2
C -> 3
...
Z -> 26
AA -> 27
AB -> 28
...


来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/excel-sheet-column-number
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn title_to_number(column_title: String) -> i32 {
    const BASE: i32 = (b'A' - 1) as i32;
    column_title
        .chars()
        .into_iter()
        .fold(0, |b, e| b * 26 + (e as i32 - BASE))
}

#[inject_description(
    problems = "PROBLEMS",
    id = "190",
    title = "Reverse Bits",
    topic = "algorithm",
    difficulty = "easy",
    tags = "BitManipulation, DivideAndConquer",
    note = "Reverse bits of a given 32 bits unsigned integer.

Note:

Note that in some languages, such as Java, there is no unsigned integer type. In this case, both input and output will be given as a signed integer type. They should not affect your implementation, as the integer's internal binary representation is the same, whether it is signed or unsigned.
In Java, the compiler represents the signed integers using 2's complement notation. Therefore, in Example 2 above, the input represents the signed integer -3 and the output represents the signed integer -1073741825.
 



来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/reverse-bits
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn reverse_bits(x: u32) -> u32 {
    x.reverse_bits()
}

#[inject_description(
    problems = "PROBLEMS",
    id = "191",
    title = "Number of 1 Bits",
    topic = "algorithm",
    difficulty = "easy",
    tags = "BitManipulation",
    note = "Write a function that takes an unsigned integer and returns the number of '1' bits it has (also known as the Hamming weight).

Note:

Note that in some languages, such as Java, there is no unsigned integer type. In this case, the input will be given as a signed integer type. It should not affect your implementation, as the integer's internal binary representation is the same, whether it is signed or unsigned.
In Java, the compiler represents the signed integers using 2's complement notation. Therefore, in Example 3, the input represents the signed integer. -3.


来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/number-of-1-bits
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn hamming_weight(n: u32) -> i32 {
    n.count_ones() as i32
}

#[inject_description(
    problems = "PROBLEMS",
    id = "202",
    title = "Happy Number",
    topic = "algorithm",
    difficulty = "easy",
    tags = "HashTable,Math,TwoPointers",
    note = "Write an algorithm to determine if a number n is happy.

A happy number is a number defined by the following process:

Starting with any positive integer, replace the number by the sum of the squares of its digits.
Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
Those numbers for which this process ends in 1 are happy.
Return true if n is a happy number, and false if not.

 

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/happy-number
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn is_happy(n: i32) -> bool {
    let (mut buf, mut history) = (vec![], HashSet::new());
    let mut n = n as u64;

    while n != 0 && !history.contains(&n) {
        if n == 1 {
            return true;
        }
        history.insert(n);

        buf.clear();
        while n != 0 {
            let pow = (n % 10) * (n % 10);
            n /= 10;
            buf.push(pow)
        }
        n = buf.iter().sum();
    }

    false
}

#[inject_description(
    problems = "PROBLEMS",
    id = "203",
    title = "Remove Linked List Elements",
    topic = "algorithm",
    difficulty = "easy",
    tags = "Recursion, LinkedList",
    note = "Given the head of a linked list and an integer val, remove all the nodes of the linked list that has Node.val == val, and return the new head.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/remove-linked-list-elements
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn remove_elements(mut head: Option<Box<ListNode>>, val: i32) -> Option<Box<ListNode>> {
    let mut fake_head = Box::new(ListNode::new(0));
    let mut tail = &mut fake_head;
    while let Some(mut node) = head {
        head = node.next.take();
        if node.val != val {
            tail.next.replace(node);
            tail = tail.next.as_mut().unwrap_or_else(|| unreachable!());
        }
    }

    fake_head.next
}

#[inject_description(
    problems = "PROBLEMS",
    id = "205",
    title = "Isomorphic Strings",
    topic = "algorithm",
    difficulty = "easy",
    tags = "HashTable, string",
    note = "Given two strings s and t, determine if they are isomorphic.

Two strings s and t are isomorphic if the characters in s can be replaced to get t.

All occurrences of a character must be replaced with another character while preserving the order of characters. No two characters may map to the same character, but a character may map to itself.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/isomorphic-strings
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn is_isomorphic(s: String, t: String) -> bool {
    if s.len() != t.len() {
        false
    } else {
        let mut m: HashMap<char, char> = HashMap::new();
        for (a, b) in s.chars().zip(t.chars()) {
            match m.get(&a) {
                None => {
                    m.insert(a, b);
                }
                Some(&x) => {
                    if x != b {
                        return false;
                    }
                }
            }
        }

        true
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "206",
    title = "Reverse Linked List",
    topic = "algorithm",
    difficulty = "easy",
    tags = "Recursion, LinkedList",
    note = "Given the head of a singly linked list, reverse the list, and return the reversed list.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/reverse-linked-list
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn reverse_list(mut head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut fake_head = Box::new(ListNode::new(0));

    while let Some(mut node) = head {
        head = node.next.take();
        let tail = fake_head.next.take();
        node.next = tail;
        fake_head.next.replace(node);
    }

    fake_head.next
}

#[inject_description(
    problems = "PROBLEMS",
    id = "217",
    title = "Contains Duplicate",
    topic = "algorithm",
    difficulty = "easy",
    tags = "array,hashtable, sorting",
    note = "Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/contains-duplicate
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn contains_duplicate(mut nums: Vec<i32>) -> bool {
    nums.sort_unstable();
    nums.iter()
        .zip(nums.iter().skip(1))
        .any(|(&pre, &fol)| pre == fol)
}

#[inject_description(
    problems = "PROBLEMS",
    id = "219",
    title = "Contains Duplicate II",
    topic = "algorithm",
    difficulty = "easy",
    tags = "Array, hashTable, slidingWindow",
    note = "Given an integer array nums and an integer k, return true if there are two distinct indices i and j in the array such that nums[i] == nums[j] and abs(i - j) <= k.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/contains-duplicate-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn contains_nearby_duplicate(nums: Vec<i32>, k: i32) -> bool {
    let mut x = nums
        .into_iter()
        .enumerate()
        .map(|(i, x)| (i, x))
        .collect::<Vec<_>>();
    x.sort_unstable_by_key(|e| e.1);

    x.iter()
        .zip(x.iter().skip(1))
        .any(|(pre, fol)| pre.1 == fol.1 && fol.0.abs_diff(pre.0) <= k as usize)
}

#[inject_description(
    problems = "PROBLEMS",
    id = "226",
    title = "Invert Binary Tree",
    topic = "algorithm",
    difficulty = "easy",
    tags = "Tree, DepthFirstSearch, BreadthFirstSearch, BinaryTree",
    note = "Given the root of a binary tree, invert the tree, and return its root.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/invert-binary-tree
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn invert_tree(root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
    let (res, mut stk) = match root {
        None => {
            return None;
        }
        Some(node) => (Some(node.clone()), vec![node]),
    };

    while let Some(node) = stk.pop() {
        let left = node.borrow_mut().left.take();
        let right = node.borrow_mut().right.take();

        if let Some(l) = left {
            stk.push(l.clone());
            node.borrow_mut().right.replace(l);
        }

        if let Some(r) = right {
            stk.push(r.clone());
            node.borrow_mut().left.replace(r);
        }
    }

    res
}
