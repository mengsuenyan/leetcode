//! BTree相关

use std::cell::{RefCell};
use std::rc::Rc;
use std::collections::VecDeque;
use crate::lc_list::ListNode;

#[derive(Eq, PartialEq, Debug)]
pub struct TreeNode {
    pub val: i32,
    pub left: Option<Rc<RefCell<TreeNode>>>,
    pub right: Option<Rc<RefCell<TreeNode>>>,
}

impl TreeNode {
    pub fn new(val: i32) -> Self {
        TreeNode {
            val,
            left: None,
            right: None,
        }
    }
}

type TreeRoot = Option<Rc<RefCell<TreeNode>>>;
impl TreeNode {
    /// Binary Tree Preorder Traversal
    /// 
    /// Given a binary tree, return the preorder traversal of its nodes’ values.
    pub fn preorder_traversal(root: TreeRoot) -> Vec<i32> {
        let mut stk = Vec::with_capacity(128);
        let mut res = Vec::with_capacity(128);
        if root.is_some() {stk.push(root.unwrap().clone());}
        
        while let Some(x) = stk.pop() {
            res.push(x.borrow().val);
            if x.borrow().right.is_some() {
                stk.push(x.borrow().right.clone().unwrap());
            }
            if x.borrow().left.is_some() {
                stk.push(x.borrow().left.clone().unwrap());
            }
        }
        
        res
    }
    
    /// Binary Tree Inorder Traversal
    pub fn inorder_traversal(root: TreeRoot) -> Vec<i32> {
        let mut stk = Vec::with_capacity(128);
        let mut res = Vec::with_capacity(128);
        
        let mut p = root;
        
        while !stk.is_empty() || p.is_some() {
            match p {
                Some(x) => {
                    p = x.borrow().left.clone();
                    stk.push(x);
                },
                None => {
                    p = stk.pop();
                    let tmp = p.as_ref().unwrap().clone();
                    res.push(tmp.borrow().val);
                    p = tmp.borrow().right.clone();
                }
            }
        }
        
        res
    }
    
    /// Binary Tree Postorder Traversal
    pub fn postorder_traversal(root: TreeRoot) -> Vec<i32> {
        let mut stk = Vec::with_capacity(128);
        let mut res = Vec::with_capacity(128);
        
        let mut p = root;
        
        loop {
            while let Some(x) = p {
                p = x.borrow().left.clone();
                stk.push(x);
            }
            
            let mut q =  None;
            while let Some(x) = stk.pop() {
                p = Some(x.clone());
                if x.borrow().right == q {
                    // 右孩子已经访问过/不存在
                    res.push(x.borrow().val);
                    q = Some(x);
                } else {
                    // 开始处理右子树
                    p = x.borrow().right.clone();
                    stk.push(x);
                    break;
                }
            }
            
            if stk.is_empty() {
                break;
            }
        }
        
        res
    }
    
    /// Binary Tree Level Order Traversal
    ///
    /// Given a binary tree, return the level order traversal of its nodes’ values. (ie, from le to right, level by
    /// level)
    pub fn level_order(root: TreeRoot) -> Vec<Vec<i32>> {
        if root.is_none() {return vec![];}
        let mut res = Vec::new();
        let mut level = Vec::new();
        
        let (mut next, mut cur) = (VecDeque::new(), VecDeque::new());
        let (mut pnext, mut pcur) = (&mut next, &mut cur);
        pcur.push_back(root.unwrap());
        while !pcur.is_empty() {
            while let Some(x) = pcur.pop_front() {
                level.push(x.borrow().val);
                if x.borrow().left.is_some() {
                    pnext.push_back(x.borrow().left.clone().unwrap());
                }
                if x.borrow().right.is_some() {
                    pnext.push_back(x.borrow().right.clone().unwrap());
                }
            }
            res.push(level.clone());
            level.clear();
            std::mem::swap(&mut pcur, &mut pnext);
        }
        
        res
    }
    
    /// Binary Tree Zigzag Level Order Traversal
    /// 
    /// Given a binary tree, return the zigzag level order traversal of its nodes’ values. (ie, from le to right,
    /// then right to le for the next level and alternate between)
    pub fn zigzag_level_order(root: TreeRoot) -> Vec<Vec<i32>> {
        let (mut stk, mut res) = (std::collections::VecDeque::new(), Vec::new());
        let mut is_l2r = true;
        let mut lvl = Vec::new();
        
        if root.is_none() {
            return Vec::new();
        }
        
        stk.push_back(root);
        stk.push_back(None);
        while let Some(x) = stk.pop_front() {
            match x {
                Some(node) => {
                    lvl.push(node.borrow().val);
                    if node.borrow().left.is_some() {
                        stk.push_back(node.borrow().left.clone());
                    }
                    if node.borrow().right.is_some() {
                        stk.push_back(node.borrow().right.clone());
                    }
                },
                None => {
                    if is_l2r {
                        res.push(lvl.to_vec());
                    } else {
                        lvl.reverse();
                        res.push(lvl.to_vec());
                    }

                    lvl.clear();
                    is_l2r = !is_l2r;
                    if !stk.is_empty() {
                        stk.push_back(None);
                    }
                }
            }
        }

        res
    }
    
    /// Recover Binary Search Tree
    /// Two elements of a binary search tree (BST) are swapped by mistake.
    /// Recover the tree without changing its structure.
    /// Note: A solution using O(n) space is prey straight forward. Could you devise a constant space
    /// solution?
    pub fn recover_tree(root: &mut TreeRoot) {
        let detect = |broken: &mut (TreeRoot, TreeRoot), prev: &mut TreeRoot, cur: &mut TreeRoot| {
            if prev.is_some() && prev.as_ref().unwrap().borrow().val > cur.as_ref().unwrap().borrow().val {
                if broken.0.is_none() {
                    broken.0 = prev.clone();
                }
                broken.1 = cur.clone();
            }
        };
        
        if root.is_none() {
            return;
        }

        let mut broken = (None, None);
        let (mut prev, mut cur) = (None, root.clone());
        // 先左子树, 后右子树
        while cur.is_some() {
            if cur.as_ref().unwrap().borrow().left.is_none() {
                detect(&mut broken, &mut prev, &mut cur);
                prev = cur.clone();
                let tmp = cur.as_ref().unwrap().borrow().right.clone();
                cur = tmp;
            } else {
                let mut node = cur.as_ref().unwrap().borrow().left.clone();
                while node.as_ref().unwrap().borrow().right.is_some() && node.as_ref().unwrap().borrow().right != cur {
                    let tmp = node.as_ref().unwrap().borrow().right.clone();
                    node = tmp;
                }
                
                if node.as_ref().unwrap().borrow().right.is_none() {
                    node.as_ref().unwrap().as_ref().borrow_mut().right = cur.clone();
                    let tmp = cur.as_ref().unwrap().borrow().left.clone();
                    cur = tmp;
                } else {
                    detect(&mut broken, &mut prev, &mut cur);
                    node.as_ref().unwrap().as_ref().borrow_mut().right = None;
                    prev = cur.clone();
                    let tmp = cur.as_ref().unwrap().borrow().right.clone();
                    cur = tmp;
                }
            }
        }
        
        std::mem::swap(&mut broken.0.as_mut().unwrap().as_ref().borrow_mut().val, 
                        &mut broken.1.as_mut().unwrap().as_ref().borrow_mut().val);
    }
    
    /// Same Tree
    /// Given two binary trees, write a function to check if they are equal or not.
    /// Two binary trees are considered equal if they are structurally identical and the nodes have the same
    /// value.
    pub fn is_same_tree(p: TreeRoot, q: TreeRoot) -> bool {
        if p.is_none() && q.is_none() {
            true
        } else if p.is_none() || q.is_none() {
            false
        } else {
            (p.as_ref().unwrap().borrow().val == q.as_ref().unwrap().borrow().val) 
                && Self::is_same_tree(p.as_ref().unwrap().borrow().left.clone(), q.as_ref().unwrap().borrow().left.clone())
                && Self::is_same_tree(p.as_ref().unwrap().borrow().right.clone(), q.as_ref().unwrap().borrow().right.clone())
        } 
    }
    
    /// Symmetric Tree
    pub fn is_symmetric(root: TreeRoot) -> bool {
        let mut stk = Vec::new();
        
        if root.is_none() {return true;}
        
        stk.push(root.as_ref().unwrap().borrow().left.clone());
        stk.push(root.as_ref().unwrap().borrow().right.clone());
        
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
                },
                (Some(None), Some(None)) => {},
                _ => {return false;}
            }
        }
        
        true
    }
    
    /// Balanced Binary Tree
    /// Given a binary tree, determine if it is height-balanced.
    /// For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the
    /// two subtrees of every node never differ by more than 1
    pub fn is_balanced(root: TreeRoot) -> bool {
        Self::is_balanced_help(root) >= 0
    }
    
    fn is_balanced_help(root: TreeRoot) -> i32 {
        if root.is_none() {
            0
        } else {
            let left = Self::is_balanced_help(root.as_ref().unwrap().borrow().left.clone());
            let right = Self::is_balanced_help(root.as_ref().unwrap().borrow().right.clone());
            
            if left < 0 || right < 0 || (left - right).abs() > 1 {
                -1
            } else {
                std::cmp::max(left, right) + 1
            }
        }
    }
    
    /// Flatten Binary Tree to Linked List
    /// Given a binary tree, flaen it to a linked list in-place.
    /// For example, Given
    /// 1
    /// / \
    /// 2 5
    /// / \ \
    /// 3 4 6
    /// e flaened tree should look like:
    /// 1
    /// \
    /// 2
    /// \
    /// 3
    /// \
    /// 4
    /// \
    /// 5
    /// \
    /// 6
    pub fn flatten(root: &mut TreeRoot) {
        if root.is_none() {return;}
        
        let mut stk = Vec::new();
        stk.push(root.clone());
        
        while let Some(x) = stk.pop() {
            let x = x.unwrap();
            
            if x.borrow().right.is_some() {
                stk.push(x.borrow().right.clone());
            }
            if x.borrow().left.is_some() {
                stk.push(x.borrow().left.clone());
            }
            
            x.borrow_mut().left = None;
            if !stk.is_empty() {
                x.borrow_mut().right = stk.last().unwrap().clone();
            }
        }
    }

    /// Construct Binary Tree from Preorder and Inorder Traversal
    /// Given preorder and inorder traversal of a tree, construct the binary tree.
    /// Note: You may assume that duplicates do not exist in the tree.
    pub fn build_tree_from_pre_in(preorder: Vec<i32>, inorder: Vec<i32>) -> TreeRoot {
        Self::build_tree_help(&preorder, (0, preorder.len()), &inorder, (0, inorder.len()))
    }
    
    fn build_tree_help(preorder: &Vec<i32>, po: (usize, usize), inorder: &Vec<i32>, io: (usize, usize)) -> TreeRoot {
        if po.0 >= po.1 { return None; }
        if io.0 >= io.1 {return None; }
        
        let mut node = TreeNode::new(preorder[po.0]);
        let len = match inorder.iter().skip(io.0).take(io.1 - io.0).position(|&x| {
            preorder[po.0] == x
        }) {
            Some(idx) => {idx},
            None => {io.1 - io.0},
        };
        
        node.left = Self::build_tree_help(preorder, (po.0 + 1, po.0 + 1 + len),
            inorder, (io.0, io.0 + len));
        node.right = Self::build_tree_help(preorder, (po.0 + len + 1, po.1),
            inorder, (io.0 + len + 1, io.1));
        
        Some(Rc::new(RefCell::new(node)))
    }
    
    /// Construct Binary Tree from Inorder and Postorder Traversal
    pub fn build_tree_from_in_post(inorder: Vec<i32>, postorder: Vec<i32>) -> TreeRoot {
        Self::build_tree_help_ii(&inorder, (0, inorder.len()), &postorder, (0, postorder.len()))
    }
    
    fn build_tree_help_ii(inorder: &Vec<i32>, io: (usize, usize), postorder: &Vec<i32>, po: (usize, usize)) -> TreeRoot {
        if io.0 >= io.1 || po.0 >= po.1 { return None; }
        
        let mut node = TreeNode::new(postorder[po.1 - 1]);
        let len = match inorder.iter().skip(io.0).take(io.1 - io.0).position(|&x| {
            postorder[po.1 - 1] == x
        }) {
            Some(idx) => {idx},
            None => {io.1 - io.0},
        };

        node.left = Self::build_tree_help_ii(inorder, (io.0, io.0 + len), postorder, (po.0, po.0 + len));
        node.right = Self::build_tree_help_ii(inorder, (io.0 + len + 1, io.1), postorder, (po.0 + len, po.1 - 1));
        
        Some(Rc::new(RefCell::new(node)))
    }
    
    /// Unique Binary Sear Trees
    /// Given n, how many structurally unique BST’s (binary search trees) that store values 1:::n?
    /// For example, Given n = 3, there are a total of 5 unique BST’s.
    /// 1 3 3 2 1
    /// \ / / / \ \
    /// 3 2 1 1 3 2
    /// / / \ \
    /// 2 1 2 3
    pub fn num_trees(n: i32) -> i32 {
        // $\sum_{k=1}^{i} f(k-1) * f(i-k)$
        debug_assert!(n > 0);
        let n = n as usize;
        let mut f = Vec::new();
        f.resize(n+1, 0);
        f[0] = 1;
        f[1] = 1;
        
        for i in 2..=n {
            for k in 1..=i {
                f[i] += f[k-1] * f[i-k];
            }
        }
        
        f[n]
    }
    
    /// Unique Binary Search Trees II
    pub fn generate_trees(n: i32) -> Vec<TreeRoot> {
        if n > 0 {
            Self::generate_trees_help(1,n)
        } else {
            Vec::new()
        }
    }
    
    fn generate_trees_help(start: i32, end: i32) -> Vec<TreeRoot> {
        let mut trees = Vec::new();
        if start > end {
            trees.push(None);
            return trees;
        }
        
        for k in start..=end {
            let left_trees = Self::generate_trees_help(start, k - 1);
            let right_trees = Self::generate_trees_help(k + 1, end);
            for i in left_trees.iter() {
                for j in right_trees.iter() {
                    let mut node = TreeNode::new(k);
                    node.left = i.clone();
                    node.right = j.clone();
                    trees.push(Some(Rc::new(RefCell::new(node))));
                }
            }
        }
        
        trees
    }
    
    /// Validate Binary Search Tree
    pub fn is_valid_bst(root: TreeRoot) -> bool {
        Self::is_valid_bst_help(root, None, None)
    }
    
    fn is_valid_bst_help(root: TreeRoot, lower: Option<i32>, upper: Option<i32>) -> bool {
        match root {
            Some(x) => {
                (match (lower, upper) {
                    (Some(l), Some(u)) => { x.borrow().val > l && x.borrow().val < u},
                    (Some(l), None) => {x.borrow().val > l},
                    (None, Some(u)) => {x.borrow().val < u},
                    _ => {true},
                }) && Self::is_valid_bst_help(x.borrow().left.clone(), lower, Some(x.borrow().val)) &&
                    Self::is_valid_bst_help(x.borrow().right.clone(), Some(x.borrow().val), upper)
            },
            None => {
                true
            },
        }
    }
    
    /// Convert Sorted Array to Binary Search Tree
    pub fn sorted_array_to_bst(num: Vec<i32>) -> TreeRoot {
        Self::sorted_array_to_bst_help(&num, 0, num.len())
    }
    
    fn sorted_array_to_bst_help(num: &Vec<i32>, start: usize, end: usize) -> TreeRoot {
        if start >= end {
            return None;
        }
        
        let mid = start + ((end - start) >> 1);
        let mut node = TreeNode::new(num[mid]);
        node.left = Self::sorted_array_to_bst_help(num, start, mid);
        node.right = Self::sorted_array_to_bst_help(num, mid + 1, end);
        
        Some(Rc::new(RefCell::new(node)))
    }
    
    /// Convert Sorted ListNode to Binary Search Tree
    pub fn sorted_list_to_bst(head: Option<Box<ListNode>>) -> TreeRoot {
        let nums = ListNode::to_vec(&head);
        
        Self::sorted_array_to_bst(nums)
    }
    
    /// Minimum Depth of Binary Tree
    /// Given a binary tree, find its minimum depth.
    /// e minimum depth is the number of nodes along the shortest path from the root node down to the
    /// nearest leaf node.
    pub fn min_depth(root: TreeRoot) -> i32 {
        Self::min_depth_help(root, false)
    }
    
    fn min_depth_help(root: TreeRoot, is_has_brother: bool) -> i32 {
        match root {
            Some(x) => {
                1 + std::cmp::min(Self::min_depth_help(x.borrow().left.clone(), x.borrow().right.is_some()),
                    Self::min_depth_help(x.borrow().right.clone(), x.borrow().left.is_some()))
            },
            None => {
                if is_has_brother {std::i32::MAX} else {0}
            }
        }
    }
    
    pub fn max_depth(root: TreeRoot) -> i32 {
        match root {
            Some(x) => {
                std::cmp::max(Self::max_depth(x.borrow().left.clone()), Self::max_depth(x.borrow().right.clone())) + 1
            },
            None => 0,
        }
    }

    /// Path Sum
    /// Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that adding up all the
    /// values along the path equals the given sum
    pub fn has_path_sum(root: TreeRoot, sum: i32) -> bool {
        match root {
            Some(x) => {
                if x.borrow().left.is_none() && x.borrow().right.is_none() {
                    x.borrow().val == sum
                } else {
                    Self::has_path_sum(x.borrow().left.clone(), sum - x.borrow().val) ||
                        Self::has_path_sum(x.borrow().right.clone(), sum - x.borrow().val)
                }
            },
            None => {
                false
            }
        }
    }
    
    /// Path Sum
    /// Given a binary tree and a sum, find all root-to-leaf paths where each path’s sum equals the given sum.
    pub fn path_sum(root: TreeRoot, sum: i32) -> Vec<Vec<i32>> {
        let (mut cur, mut res): (Vec<i32>, Vec<Vec<i32>>) = (Vec::new(), Vec::new());
        Self::path_sum_help(root, sum, &mut cur, &mut res);
        res
    }
    
    fn path_sum_help(root: TreeRoot, gap: i32, cur: &mut Vec<i32>, res: &mut Vec<Vec<i32>>) {
        match root {
            Some(x) => {
                cur.push(x.borrow().val);
                if x.borrow().left.is_none() && x.borrow().right.is_none() && gap == x.borrow().val {
                    res.push(cur.to_vec())
                }
                Self::path_sum_help(x.borrow().left.clone(), gap - x.borrow().val, cur, res);
                Self::path_sum_help(x.borrow().right.clone(), gap - x.borrow().val, cur, res);
                cur.pop();
            },
            None => {},
        }
    }
    
    /// Binary Tree Maximum Path Sum
    /// Given a binary tree, find the maximum path sum.
    /// e path may start and end at any node in the tree. For example: Given the below binary tree,
    /// 1
    /// / \
    /// 2 3
    /// Return 6
    pub fn max_path_sum(root: TreeRoot) -> i32 {
        let mut max_sum = std::i32::MIN;
        Self::max_path_sum_help(root, &mut max_sum);
        max_sum
    }
    
    fn max_path_sum_help(root: TreeRoot, max_sum: &mut i32) -> i32 {
        match root {
            Some(x) => {
                let (l, r) = (Self::max_path_sum_help(x.borrow().left.clone(), max_sum), 
                    Self::max_path_sum_help(x.borrow().right.clone(), max_sum));
                let mut sum = x.borrow().val;
                if l > 0 {sum += l;}
                if r > 0 {sum += r;}
                *max_sum = std::cmp::max(*max_sum, sum);
                if std::cmp::max(l, r) > 0 {
                    std::cmp::max(l, r) + x.borrow().val
                } else {
                    x.borrow().val
                }
            },
            None => {
                0
            }
        }
    }
    
    /// Sum Root to Leaf Numbers
    /// Given a binary tree containing digits from 0-9 only, each root-to-leaf path could represent a number.
    /// An example is the root-to-leaf path 1->2->3 which represents the number 123.
    /// Find the total sum of all root-to-leaf numbers.
    /// For example,
    /// 1
    /// / \
    /// 2 3
    /// e root-to-leaf path 1->2 represents the number 12. e root-to-leaf path 1->3 represents the number 13.
    /// Return the sum = 12 + 13 = 25.
    pub fn sum_numbers(root: TreeRoot) -> i32 {
        Self::sum_numbers_help(root, 0)
    }
    
    fn sum_numbers_help(root: TreeRoot, sum: i32) -> i32 {
        match root {
            Some(x) => {
                if x.borrow().left.is_none() && x.borrow().right.is_none() {
                    sum * 10 + x.borrow().val
                } else {
                    Self::sum_numbers_help(x.borrow().left.clone(), sum * 10 + x.borrow().val) +
                        Self::sum_numbers_help(x.borrow().right.clone(), sum * 10 + x.borrow().val)
                }
            },
            None => {
                0
            }
        }
    }
}
