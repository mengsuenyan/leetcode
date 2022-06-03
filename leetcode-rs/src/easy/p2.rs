use std::cell::{Cell, RefCell};
use crate::easy::p1::TreeNode;
use crate::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::sync::{Arc, RwLock};

lazy_static::lazy_static! {
    pub static ref PROBLEMS: Arc<RwLock<Problems<Problem>>> =Arc::new(RwLock::new(Problems::new()));
}

#[inject_description(
    problems = "PROBLEMS",
    id = "101",
    title = "Symmetric Tree",
    topic = "algorithm",
    difficulty = "easy",
    tags = "tree, DepthFirstSearch, BreadthFirstSearch, BinaryTree",
    note = "Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/symmetric-tree
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn is_symmetric(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
    let mut stk = Vec::new();
    match root.as_ref() {
        None => {
            return true;
        }
        Some(node) => {
            stk.push(node.borrow().left.clone());
            stk.push(node.borrow().right.clone());
        }
    }

    while !stk.is_empty() {
        match (stk.pop(), stk.pop()) {
            (Some(Some(p)), Some(Some(q))) => {
                if p.borrow().val != q.borrow().val {
                    return false;
                } else {
                    stk.push(p.borrow().left.clone());
                    stk.push(q.borrow().right.clone());
                    stk.push(p.borrow().right.clone());
                    stk.push(q.borrow().left.clone());
                }
            }
            (Some(None), Some(None)) => {return true;}
            _ => {return false;}
        }
    }

    true
}

#[inject_description(
problems = "PROBLEMS",
id = "104",
title = "Maximum Depth of Binary Tree",
topic = "algorithm",
difficulty = "easy",
tags = "Tree, DepthFirstSearch, BreathFirstSearch, BinaryTree",
note = "Given the root of a binary tree, return its maximum depth.

A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/maximum-depth-of-binary-tree
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。",
)]
pub fn max_depth(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    match root {
        None => {return 0;}
        Some(r) => {
            let (mut max_depth, mut stk) = (0, Vec::new());
            stk.push((r, 1));

            while let Some((node, depth)) = stk.pop() {
                if depth > max_depth {
                    max_depth = depth;
                }

                if let Some(left) = node.borrow().left.as_ref() {
                    stk.push((left.clone(), depth + 1))
                }

                if let Some(right) = node.borrow().right.as_ref() {
                    stk.push((right.clone(), depth + 1))
                }
            }

            max_depth
        }
    }
}

#[inject_description(
problems = "PROBLEMS",
id = "108",
title = "Convert Sorted Array to Binary Search Tree",
topic = "algorithm",
difficulty = "easy",
tags = "Tree, BinarySearchTree, Array, DivideAndConquer, BinaryTree",
note = "Given an integer array nums where the elements are sorted in ascending order, convert it to a height-balanced binary search tree.

A height-balanced binary tree is a binary tree in which the depth of the two subtrees of every node never differs by more than one.



来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。",
)]
pub fn sorted_array_to_bst(nums: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
    fn to_bst(nums: &[i32]) -> Option<Rc<RefCell<TreeNode>>> {
        if nums.is_empty() {
            None
        } else {
            let mid = nums.len() >> 1;
            let mut node = TreeNode::new(nums[mid]);
            node.left = to_bst(&nums[0..mid]);
            node.right = to_bst(&nums[(mid + 1)..]);
            Some(Rc::new(RefCell::new(node)))
        }
    }

    to_bst(nums.as_slice())
}

#[inject_description(
problems = "PROBLEMS",
id = "110",
title = "Balanced Binary Tree",
topic = "algorithm",
difficulty = "easy",
tags = "Tree,DepthFirstSearch, BinaryTree",
note = "Given a binary tree, determine if it is height-balanced.

For this problem, a height-balanced binary tree is defined as:

a binary tree in which the left and right subtrees of every node differ in height by no more than 1.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/balanced-binary-tree
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。",
)]
pub fn is_balanced(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
    let (mut min, mut max, mut stk) = (i32::MAX, 0, VecDeque::new());

    match root {
        None => {return true;}
        Some(node) => {
            stk.push_back((node, 1));
        }
    }

    while let Some((node, level)) = stk.pop_front() {
        if level > max {
            max = level;
        }

        match node.borrow().left.clone() {
            None => {
                if level < min {
                    min = level;
                }
            }
            Some(left) => {
                stk.push_back((left, level + 1))
            }
        }

        match node.borrow().right.clone() {
            None => {
                if level < min {
                    min = level;
                }
            }
            Some(right) => {
                stk.push_back((right, level + 1));
            }
        }
    }

    (max - min) <= 1
}

#[inject_description(
problems = "PROBLEMS",
id = "111",
title = "Minimum Depth of Binary Tree",
topic = "algorithm",
difficulty = "easy",
tags = "Tree, DepthFirstSearch, BreadthFirstSearch, BinaryTree",
note = "Given a binary tree, find its minimum depth.

The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.

Note: A leaf is a node with no children.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/minimum-depth-of-binary-tree
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。",
)]
pub fn min_depth(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let (mut min, mut stk) = (i32::MAX, VecDeque::new());
    match root {
        None => {return 0;}
        Some(node) => {
            stk.push_back((node, 1));
        }
    }

    while let Some((node, lvl)) = stk.pop_front() {
        match (node.borrow().left.clone(), node.borrow().right.clone()) {
            (None, None) => {
                if lvl < min {
                    min = lvl;
                }
            },
            (Some(left), None) => {
                stk.push_back((left, lvl + 1));
            },
            (None, Some(right)) => {
                stk.push_back((right, lvl + 1));
            },
            (Some(left), Some(right)) => {
                stk.push_back((left, lvl + 1));
                stk.push_back((right, lvl + 1));
            }
        }
    }

    min
}

#[inject_description(
problems = "PROBLEMS",
id = "112",
title = "Path Sum",
topic = "algorithm",
difficulty = "easy",
tags = "Tree, DepthFirstSearch, BreadthFirstSearch, BinaryTree",
note = "Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path such that adding up all the values along the path equals targetSum.

A leaf is a node with no children.

Constraints:
The number of nodes in the tree is in the range [0, 5000].
-1000 <= Node.val <= 1000
-1000 <= targetSum <= 1000

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/path-sum
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。",
)]
pub fn has_path_sum(root: Option<Rc<RefCell<TreeNode>>>, target_sum: i32) -> bool {
    let mut stk = match root {
        None => {return false;}
        Some(node) => {
            let (mut stk, sum) = (VecDeque::new(), node.borrow().val);
            stk.push_back((sum, node));
            stk
        }
    };

    while let Some((sum, node)) = stk.pop_front() {
        match (node.borrow().left.clone(), node.borrow().right.clone()) {
            (None, None) => {
                if sum == target_sum {
                    return true;
                }
            },
            (Some(left), None) => {
                let lsum = left.borrow().val + sum;
                stk.push_front((lsum, left));
            },
            (None, Some(right)) => {
                let rsum = right.borrow().val + sum;
                stk.push_front((rsum, right));
            },
            (Some(left), Some(right)) => {
                let lsum = right.borrow().val + sum;
                stk.push_front((lsum, right));
                let rsum = left.borrow().val + sum;
                stk.push_front((rsum, left));
            }
        }
    }

    false
}

#[inject_description(
problems = "PROBLEMS",
id = "118",
title = "Pascal's Triangle",
topic = "algorithm",
difficulty = "easy",
tags = "Array, DynamicProgramming",
note = "Given an integer numRows, return the first numRows of Pascal's triangle.

In Pascal's triangle, each number is the sum of the two numbers directly above it as shown:



来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/pascals-triangle
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。",
)]
pub fn generate(num_rows: i32) -> Vec<Vec<i32>> {
    (1..num_rows).fold(vec![vec![1]], |mut pascal, nums| {
        let mut row = Vec::with_capacity(nums as usize);
        row.push(1);
        for (&e1, &e2) in pascal.last().unwrap().iter().zip(pascal.last().unwrap().iter().skip(1)) {
            row.push(e1 + e2);
        }
        row.push(1);
        pascal.push(row);
        pascal
    })
}

#[inject_description(
problems = "PROBLEMS",
id = "119",
title = "Pascal's Triangle II",
topic = "algorithm",
difficulty = "easy",
tags = "Array, DynamicProgramming",
note = "Given an integer rowIndex, return the rowIndexth (0-indexed) row of the Pascal's triangle.

In Pascal's triangle, each number is the sum of the two numbers directly above it as shown:

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/pascals-triangle-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。",
)]
pub fn get_row(row_index: i32) -> Vec<i32> {
    (1..(row_index + 1)).fold((vec![1], vec![1]), |(pre, mut cur), _| {
        cur.clear();
        cur.push(1);
        for (&e1, &e2) in pre.iter().zip(pre.iter().skip(1)) {
            cur.push(e1 + e2);
        }
        cur.push(1);
        (cur, pre)
    }).0
}

#[inject_description(
problems = "PROBLEMS",
id = "121",
title = "Best Time to Buy and Sell Stock",
topic = "algorithm",
difficulty = "easy",
tags = "Array, DynamicProgramming",
note = "You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.



来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/best-time-to-buy-and-sell-stock
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。",
)]
pub fn max_profit(prices: Vec<i32>) -> i32 {
    prices.first().map(|&x| {
        prices.iter().skip(1).fold((0, x), |(profit, low_point), &val| {
            (std::cmp::max(profit, val - low_point), std::cmp::min(low_point, val))
        }).0
    }).unwrap_or_default()
}

#[inject_description(
problems = "PROBLEMS",
id = "125",
title = "Valid Palindrome",
topic = "algorithm",
difficulty = "easy",
tags = "TwoPointers, String",
note = "A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.

Given a string s, return true if it is a palindrome, or false otherwise.

Constraints:
1 <= s.length <= 2 * 10^5
s consists only of printable ASCII characters.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/valid-palindrome
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。",
)]
pub fn is_palindrome(s: String) -> bool {
    let (len, mut head, mut tail) = (s.len().saturating_sub(1), s.chars().enumerate().peekable(), s.chars().rev().enumerate().peekable());

    while let (Some(&(lpos, left)), Some(&(rpos, right))) = (head.peek(), tail.peek()) {
        if lpos >= (len - rpos) {
            break;
        }

        match (left.is_ascii_alphanumeric(), right.is_ascii_alphanumeric()) {
            (true, true) => {
                if left.to_ascii_lowercase() != right.to_ascii_lowercase() {
                    return false;
                }
                head.next();
                tail.next();
            },
            (true, false) => {
                tail.next();
            },
            (false, true) => {
                head.next();
            },
            (false, false) => {
                head.next();
                tail.next();
            }
        }
    }

    true
}

#[inject_description(
problems = "PROBLEMS",
id = "136",
title = "Single Number",
topic = "algorithm",
difficulty = "easy",
tags = "BitManipulation, Array",
note = "Given a non-empty array of integers nums, every element appears twice except for one. Find that single one.

You must implement a solution with a linear runtime complexity and use only constant extra space.
Constraints:

1 <= nums.length <= 3 * 10^4
-3 * 10^4 <= nums[i] <= 3 * 10^4
Each element in the array appears twice except for one element which appears only once.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/single-number
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。",
)]
pub fn single_number(nums: Vec<i32>) -> i32 {
    debug_assert!(!nums.is_empty(), "`nums` must be not empty");

    nums.into_iter().fold(0, |b, e| b ^ e) ^ 0
}

#[inject_description(
problems = "PROBLEMS",
id = "144",
title = "Binary Tree Preorder Traversal",
topic = "algorithm",
difficulty = "easy",
tags = "Stack, Tree, DepthFirstSearch, BinaryTree",
note = "Given the root of a binary tree, return the preorder traversal of its nodes' values.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/binary-tree-preorder-traversal
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。",
)]
pub fn preorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    let (mut stk, mut res) =  match root {
        None => {return vec![];}
        Some(root) => {
            (vec![root], vec![])
        }
    };

    while let Some(node) = stk.pop() {
        res.push(node.borrow().val);
        if let Some(right) = node.borrow().right.clone() {
            stk.push(right);
        }
        if let Some(left) = node.borrow().left.clone() {
            stk.push(left);
        }
    }

    res
}

#[inject_description(
problems = "PROBLEMS",
id = "145",
title = "Binary Tree Postorder Traversal",
topic = "algorithm",
difficulty = "easy",
tags = "stack,tree,DepthFirstSearch, binarytree",
note = "Given the root of a binary tree, return the postorder traversal of its nodes' values.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/binary-tree-postorder-traversal
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。",
)]
pub fn postorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    let (mut stk, mut res) = match root {
        None => {
            return vec![];
        },
        Some(node) => {
            (vec![node], vec![])
        }
    };

    while let Some(node) = stk.pop() {
        res.push(node.borrow().val);
        if let Some(right) = node.borrow().right.clone() {
            stk.push(right);
        }
        if let Some(left) = node.borrow().left.clone() {
            stk.push(left);
        }
    }

    res.reverse();
    res
}

#[inject_description(
problems = "PROBLEMS",
id = "155",
title = "Min Stack",
topic = "algorithm",
difficulty = "easy",
tags = "Stack, Design",
note = "Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the MinStack class:

MinStack() initializes the stack object.
void push(int val) pushes the element val onto the stack.
void pop() removes the element on the top of the stack.
int top() gets the top element of the stack.
int getMin() retrieves the minimum element in the stack.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/min-stack
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。",
)]
pub struct MinStack {
    data: Cell<Vec<i32>>,
    min: Cell<i32>,
}

#[inject_description(
problems = "PROBLEMS",
id = "155",
title = "Min Stack",
topic = "algorithm",
difficulty = "easy",
tags = "Stack, Design",
note = "Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the MinStack class:

MinStack() initializes the stack object.
void push(int val) pushes the element val onto the stack.
void pop() removes the element on the top of the stack.
int top() gets the top element of the stack.
int getMin() retrieves the minimum element in the stack.

Constraints:
-2^31 <= val <= 2^31 - 1
Methods pop, top and getMin operations will always be called on non-empty stacks.
At most 3 * 10^4 calls will be made to push, pop, top, and getMin.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/min-stack
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。",
)]
impl MinStack {
    pub fn new() -> Self {
        Self {
            data: Cell::new(Vec::new()),
            min: Cell::new(i32::MAX),
        }
    }

    pub fn push(&self, val: i32) {
        if val < self.min.get()  {
            self.min.set(val);
        }

        unsafe {
            (&mut *self.data.as_ptr()).push(val);
        }
    }

    pub fn pop(&self) {
        if let Some(val) = unsafe {
            (&mut *self.data.as_ptr()).pop()
        } {
            if val < self.min.get() {
                let x = unsafe {
                    &*self.data.as_ptr()
                }.iter().min().map(|&x| x).unwrap_or_else(|| i32::MAX);
                self.min.set(x);
            }
        }
    }

    pub fn top(&self) -> i32 {
        unsafe {&*self.data.as_ptr()}.last().map(|&x| x).unwrap_or_default()
    }

    pub fn get_min(&self) -> i32 {
        self.min.get()
    }
}

#[inject_description(
problems = "PROBLEMS",
id = "168",
title = "Excel Sheet Column Title",
topic = "algorithm",
difficulty = "easy",
tags = "Math, String",
note = "Given an integer columnNumber, return its corresponding column title as it appears in an Excel sheet.

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
链接：https://leetcode.cn/problems/excel-sheet-column-title
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。",
)]
pub fn convert_to_title(mut column_number: i32) -> String {
    std::iter::from_fn(move || {
        if column_number <= 0 {
            None
        } else {
            let n = column_number;
            column_number = (column_number - 1) / 26;
            Some(char::from(b'A' + ((n - 1) % 26) as u8))
        }
    }).collect::<Vec<_>>().into_iter().rev().collect()
}

#[inject_description(
problems = "PROBLEMS",
id = "169",
title = "Majority Element",
topic = "algorithm",
difficulty = "easy",
tags = "array,hashtable,divideandconquer,counting,sorting",
note = "Given an array nums of size n, return the majority element.

The majority element is the element that appears more than ⌊n / 2⌋ times. You may assume that the majority element always exists in the array.


来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/majority-element
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。",
)]
pub fn majority_element(nums: Vec<i32>) -> i32 {
    use std::ops::AddAssign;

    let n = nums.len() >> 1;
    let mut buf: HashMap<i32, i32> = HashMap::with_capacity((n >> 2).max(16));
    for e in nums {
        buf.entry(e).or_default().add_assign(1);
    }

    buf.into_iter().find(|x| x.1 > n as i32).unwrap().0
}