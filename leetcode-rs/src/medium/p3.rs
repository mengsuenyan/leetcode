use crate::easy::p1::{ListNode, TreeNode};
use crate::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, RwLock};

lazy_static::lazy_static! {
    pub static ref PROBLEMS: Arc<RwLock<Problems<Problem>>> =Arc::new(RwLock::new(Problems::new()));
}

#[inject_description(
    problems = "PROBLEMS",
    id = "73",
    title = "Set Matrix Zeroes",
    topic = "algorithm",
    difficulty = "medium",
    tags = "",
    note = "Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.

You must do it in place.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/set-matrix-zeroes
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
#[allow(clippy::ptr_arg)]
pub fn set_zeroes(matrix: &mut Vec<Vec<i32>>) {
    let col_len = matrix.last().map(|x| x.len()).unwrap_or_default();
    let mut col = vec![false; col_len];

    matrix.iter_mut().for_each(|line| {
        let mut is_zero = false;
        line.iter().zip(col.iter_mut()).for_each(|(&x, y)| {
            if x == 0 {
                is_zero = true;
                *y = true;
            }
        });

        if is_zero {
            line.clear();
            line.resize(col_len, 0);
        }
    });

    matrix.iter_mut().for_each(|line| {
        line.iter_mut().zip(col.iter()).for_each(|(x, &y)| {
            if y {
                *x = 0;
            }
        });
    });
}

#[inject_description(
    problems = "PROBLEMS",
    id = "74",
    title = "Search a 2D Matrix",
    topic = "algorithm",
    difficulty = "medium",
    tags = "Array, BinarySearch, Matrix",
    note = "Write an efficient algorithm that searches for a value target in an m x n integer matrix matrix. This matrix has the following properties:

Integers in each row are sorted from left to right.
The first integer of each row is greater than the last integer of the previous row.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/search-a-2d-matrix
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn search_matrix(matrix: Vec<Vec<i32>>, target: i32) -> bool {
    let col_len = matrix.last().map(|x| x.len()).unwrap_or_default();

    match col_len.checked_sub(1) {
        None => false,
        Some(last_idx) => matrix
            .binary_search_by(|x| {
                let (first, last) = (x[0], x[last_idx]);
                if target < first {
                    std::cmp::Ordering::Greater
                } else if target > last {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .map(|row_idx| matrix[row_idx].binary_search(&target).is_ok())
            .unwrap_or_default(),
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "75",
    title = "Sort Colors",
    topic = "algorithm",
    difficulty = "medium",
    tags = "Array, TwoPointers, Sorting",
    note = "Given an array nums with n objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white, and blue.

We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively.

You must solve this problem without using the library's sort function.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/sort-colors
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn sort_colors(nums: &mut Vec<i32>) {
    let mut next_idx = 0;
    (0..nums.len()).for_each(|idx| {
        if nums[idx] == 0 {
            nums.swap(idx, next_idx);
            next_idx += 1;
        }
    });

    (next_idx..nums.len()).for_each(|idx| {
        if nums[idx] == 1 {
            nums.swap(idx, next_idx);
            next_idx += 1;
        }
    });
}

#[inject_description(
    problems = "PROBLEMS",
    id = "77",
    title = "Combinations",
    topic = "algorithm",
    difficulty = "medium",
    tags = "Backtracking",
    note = "Given two integers n and k, return all possible combinations of k numbers out of the range [1, n].

You may return the answer in any order.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/combinations
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn combine(n: i32, k: i32) -> Vec<Vec<i32>> {
    debug_assert!(n > 0 && k > 0 && k <= n);
    fn dfs(res: &mut Vec<Vec<i32>>, tmp: &mut Vec<i32>, cur: usize, n: usize, k: usize) {
        if tmp.len() + n + 1 - cur >= k {
            if tmp.len() == k {
                res.push(tmp.clone());
            } else {
                tmp.push(cur as i32);
                dfs(res, tmp, cur + 1, n, k);
                tmp.pop();
                dfs(res, tmp, cur + 1, n, k);
            }
        }
    }

    let (n, k) = (n as usize, k as usize);
    let possible = ((n - k + 1)..=n).product::<usize>() / (1..=k).product::<usize>();
    let (mut res, mut tmp) = (Vec::with_capacity(possible), Vec::with_capacity(k));
    dfs(&mut res, &mut tmp, 1, n, k);
    res
}

#[inject_description(
    problems = "PROBLEMS",
    id = "78",
    title = "Subsets",
    topic = "algorithm",
    difficulty = "medium",
    tags = "BitManipulation, Array, Backtracking",
    note = "Given an integer array nums of unique elements, return all possible subsets (the power set).

The solution set must not contain duplicate subsets. Return the solution in any order.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/subsets
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn subsets(nums: Vec<i32>) -> Vec<Vec<i32>> {
    fn dfs(res: &mut Vec<Vec<i32>>, tmp: &mut Vec<i32>, sets: &[i32], cur: usize) {
        if cur == sets.len() {
            res.push(tmp.clone());
        } else {
            tmp.push(sets[cur]);
            dfs(res, tmp, sets, cur + 1);
            tmp.pop();
            dfs(res, tmp, sets, cur + 1);
        }
    }

    let (mut res, mut tmp) = (
        Vec::with_capacity(1 << nums.len()),
        Vec::with_capacity(nums.len()),
    );
    dfs(&mut res, &mut tmp, nums.as_slice(), 0);
    res
}

#[inject_description(
    problems = "PROBLEMS",
    id = "79",
    title = "Word Search",
    topic = "algorithm",
    difficulty = "medium",
    tags = "Array, Backtracking, Matrix",
    note = "Given an m x n grid of characters board and a string word, return true if word exists in the grid.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.

Constraints:
board 和 word 仅由大小写英文字母组成

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/word-search
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn exist(board: Vec<Vec<char>>, word: String) -> bool {
    #[allow(clippy::too_many_arguments)]
    fn check(
        board: &Vec<Vec<char>>,
        visited: &mut Vec<Vec<bool>>,
        i: i32,
        j: i32,
        s: &[u8],
        k: i32,
        w: i32,
        h: i32,
    ) -> bool {
        if board[i as usize][j as usize] as u8 != s[k as usize] {
            false
        } else if k as usize == s.len() - 1 {
            true
        } else {
            visited[i as usize][j as usize] = true;
            let mut result = false;
            for (first, second) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
                let (ni, nj) = (i + first, j + second);
                if ni >= 0 && ni < h && nj >= 0 && nj < w && !visited[ni as usize][nj as usize] {
                    let flag = check(board, visited, ni, nj, s, k + 1, w, h);
                    if flag {
                        result = true;
                        break;
                    }
                }
            }

            visited[i as usize][j as usize] = false;
            result
        }
    }

    let (h, w) = match board.last().map(|x| x.len()) {
        Some(w) => (board.len(), w),
        None => {
            return false;
        }
    };

    let mut visited = vec![vec![false; w]; h];
    for i in 0..h {
        for j in 0..w {
            if check(
                &board,
                &mut visited,
                i as i32,
                j as i32,
                word.as_bytes(),
                0,
                w as i32,
                h as i32,
            ) {
                return true;
            }
        }
    }

    false
}

#[inject_description(
    problems = "PROBLEMS",
    id = "80",
    title = "Remove Duplicates from Sorted Array II",
    topic = "algorithm",
    difficulty = "medium",
    tags = "",
    note = "Given an integer array nums sorted in non-decreasing order, remove some duplicates in-place such that each unique element appears at most twice. The relative order of the elements should be kept the same.

Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.

Return k after placing the final result in the first k slots of nums.

Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.

Custom Judge:

The judge will test your solution with the following code:

```cpp
int[] nums = [...]; // Input array
int[] expectedNums = [...]; // The expected answer with correct length

int k = removeDuplicates(nums); // Calls your implementation

assert k == expectedNums.length;
for (int i = 0; i < k; i++) {
    assert nums[i] == expectedNums[i];
}
```
If all assertions pass, then your solution will be accepted.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/remove-duplicates-from-sorted-array-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn remove_duplicates(nums: &mut Vec<i32>) -> i32 {
    let mut cnt = 1;
    nums.dedup_by(|a, b| {
        cnt = if a == b { cnt + 1 } else { 1 };
        cnt > 2
    });

    nums.len() as i32
}

#[inject_description(
    problems = "PROBLEMS",
    id = "81",
    title = "Search in Rotated Sorted Array II",
    topic = "algorithm",
    difficulty = "medium",
    tags = "Array, BinarySearch",
    note = "There is an integer array nums sorted in non-decreasing order (not necessarily with distinct values).

Before being passed to your function, nums is rotated at an unknown pivot index k (0 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,4,4,5,6,6,7] might be rotated at pivot index 5 and become [4,5,6,6,7,0,1,2,4,4].

Given the array nums after the rotation and an integer target, return true if target is in nums, or false if it is not in nums.

You must decrease the overall operation steps as much as possible.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/search-in-rotated-sorted-array-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn search(nums: Vec<i32>, target: i32) -> bool {
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

#[inject_description(
    problems = "PROBLEMS",
    id = "86",
    title = "Partition List",
    topic = "algorithm",
    difficulty = "medium",
    tags = "LinkedList, TwoPointers",
    note = "Given the head of a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.

You should preserve the original relative order of the nodes in each of the two partitions.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/partition-list
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn partition(mut head: Option<Box<ListNode>>, x: i32) -> Option<Box<ListNode>> {
    let (mut first, mut second) = (vec![], vec![]);
    while let Some(mut node) = head {
        head = node.next.take();
        if node.val < x {
            first.push(Some(node));
        } else {
            second.push(Some(node))
        }
    }

    first.append(&mut second);
    while let Some(mut node) = first.pop() {
        if let Some(x) = node.as_mut() {
            x.next = head;
        }
        head = node;
    }

    head
}

#[inject_description(
    problems = "PROBLEMS",
    id = "89",
    title = "Gray Code",
    topic = "algorithm",
    difficulty = "medium",
    tags = "BitManipulation, Math, Backtracking",
    note = "An n-bit gray code sequence is a sequence of 2n integers where:

Every integer is in the inclusive range [0, 2n - 1],
The first integer is 0,
An integer appears no more than once in the sequence,
The binary representation of every pair of adjacent integers differs by exactly one bit, and
The binary representation of the first and last integers differs by exactly one bit.
Given an integer n, return any valid n-bit gray code sequence.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/gray-code
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn gray_code(n: i32) -> Vec<i32> {
    // g[i] = b[i+1] ^ b[i], x[i] means that the i-th bit of x
    (0..(1 << n)).fold(Vec::with_capacity(1 << n as usize), |mut res, b| {
        res.push(b ^ (b >> 1));
        res
    })
}

#[inject_description(
    problems = "PROBLEMS",
    id = "90",
    title = "Subsets II",
    topic = "algorithm",
    difficulty = "medium",
    tags = "BitManipulation, Array, Backtracking",
    note = "Given an integer array nums that may contain duplicates, return all possible subsets (the power set).

The solution set must not contain duplicate subsets. Return the solution in any order.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/subsets-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn subsets_with_dup(mut nums: Vec<i32>) -> Vec<Vec<i32>> {
    fn dfs(res: &mut Vec<Vec<i32>>, tmp: &mut Vec<i32>, nums: &[i32], cur: usize, is_pre: bool) {
        if cur == nums.len() {
            res.push(tmp.clone());
        } else {
            dfs(res, tmp, nums, cur + 1, false);
            if !(!is_pre && cur > 0 && nums[cur - 1] == nums[cur]) {
                tmp.push(nums[cur]);
                dfs(res, tmp, nums, cur + 1, true);
                tmp.pop();
            }
        }
    }

    nums.sort_unstable();
    let (mut res, mut tmp) = (vec![], vec![]);
    dfs(&mut res, &mut tmp, nums.as_slice(), 0, false);
    res
}

#[inject_description(
    problems = "PROBLEMS",
    id = "91",
    title = "Decode Ways",
    topic = "algorithm",
    difficulty = "medium",
    tags = "String, DynamicProgramming",
    note = "A message containing letters from A-Z can be encoded into numbers using the following mapping:

'A' -> \"1\"
'B' -> \"2\"
...
'Z' -> \"26\"
To decode an encoded message, all the digits must be grouped then mapped back into letters using the reverse of the mapping above (there may be multiple ways). For example, \"11106\" can be mapped into:

\"AAJF\" with the grouping (1 1 10 6)
\"KJF\" with the grouping (11 10 6)
Note that the grouping (1 11 06) is invalid because \"06\" cannot be mapped into 'F' since \"6\" is different from \"06\".

Given a string s containing only digits, return the number of ways to decode it.

The test cases are generated so that the answer fits in a 32-bit integer.

Constraints:
1 <= s.length <= 100
s contains only digits and may contain leading zero(s).

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/decode-ways
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn num_decodings(s: String) -> i32 {
    let s = s.as_bytes();
    match s.first().copied() {
        Some(b'0') => 0,
        Some(first) => {
            s.iter()
                .skip(1)
                .fold((1, 1, first), |(pre_1, pre_2, pre_char), &cur_char| {
                    // dp[i-2]
                    let cur = if cur_char == b'0' { 0 } else { pre_2 };
                    // dp[i-1]
                    let prev = if !(pre_char == b'1' || (pre_char == b'2' && cur_char <= b'6')) {
                        0
                    } else {
                        pre_1
                    };

                    // dp[i-2], dp[i-1]+dp[i-2]
                    (cur, prev + cur, cur_char)
                })
                .1
        }
        None => 0,
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "92",
    title = "Reverse Linked List II",
    topic = "algorithm",
    difficulty = "medium",
    tags = "LinkedList",
    note = "Given the head of a singly linked list and two integers left and right where left <= right, reverse the nodes of the list from position left to position right, and return the reversed list.

Constraints:
The number of nodes in the list is n.
1 <= n <= 500
-500 <= Node.val <= 500
1 <= left <= right <= n

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/reverse-linked-list-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn reverse_between(
    head: Option<Box<ListNode>>,
    left: i32,
    right: i32,
) -> Option<Box<ListNode>> {
    let (mut cnt, mut fake_head) = (0, Some(Box::new(ListNode::new(0))));
    if let Some(x) = fake_head.as_mut() {
        x.next = head;
    }

    let mut first_tail = fake_head.as_mut();

    while let Some(first) = first_tail {
        cnt += 1;
        if cnt < left {
            first_tail = first.next.as_mut();
        } else {
            let mut second_head = first.next.take();
            while let Some(mut node) = second_head {
                let nxt = node.next.take();
                node.next = first.next.take();
                first.next = Some(node);
                if cnt < right {
                    second_head = nxt;
                } else {
                    let mut second_tail = first;
                    while second_tail.next.is_some() {
                        second_tail = second_tail.next.as_mut().unwrap_or_else(|| unreachable!());
                    }
                    second_tail.next = nxt;
                    break;
                }

                cnt += 1;
            }
            break;
        }
    }

    fake_head.unwrap_or_else(|| unreachable!()).next
}

#[inject_description(
    problems = "PROBLEMS",
    id = "93",
    title = "Restore IP Addresses",
    topic = "algorithm",
    difficulty = "medium",
    tags = "String, Backtracking",
    note = "A valid IP address consists of exactly four integers separated by single dots. Each integer is between 0 and 255 (inclusive) and cannot have leading zeros.

For example, \"0.1.2.201\" and \"192.168.1.1\" are valid IP addresses, but \"0.011.255.245\", \"192.168.1.312\" and \"192.168@1.1\" are invalid IP addresses.
Given a string s containing only digits, return all possible valid IP addresses that can be formed by inserting dots into s. You are not allowed to reorder or remove any digits in s. You may return the valid IP addresses in any order.

Constraints:
1 <= s.length <= 20
s consists of digits only.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/restore-ip-addresses
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn restore_ip_addresses(s: String) -> Vec<String> {
    fn dfs(res: &mut Vec<String>, mut ip: String, s: &[u8], start: usize, step: usize) {
        if s.len() == start && step == 4 {
            ip.pop();
            res.push(ip);
        } else if (s.len() - start) >= (4 - step) && (s.len() - start) <= (4 - step) * 3 {
            let mut seg = 0;
            for (idx, &n) in s.iter().enumerate().skip(start).take(3) {
                seg = seg * 10 + (n - b'0') as usize;
                if seg <= 255 {
                    ip.push(char::from(n));
                    let mut tmp = ip.clone();
                    tmp.push('.');
                    dfs(res, tmp, s, idx + 1, step + 1);
                } else if seg == 0 {
                    // have leading zero
                    break;
                }
            }
        }
    }

    let (mut res, tmp) = (vec![], String::with_capacity(4));
    dfs(&mut res, tmp, s.as_bytes(), 0, 0);
    res
}

#[inject_description(
    problems = "PROBLEMS",
    id = "95",
    title = "Unique Binary Search Trees II",
    topic = "algorithm",
    difficulty = "medium",
    tags = "Tree, BinarySearchTree, DynamicProgramming, Backtracking, Binary",
    note = "Given an integer n, return all the structurally unique BST's (binary search trees), which has exactly n nodes of unique values from 1 to n. Return the answer in any order.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/unique-binary-search-trees-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn generate_trees(n: i32) -> Vec<Option<Rc<RefCell<TreeNode>>>> {
    fn generate_tree(start: i32, end: i32) -> Vec<Option<Rc<RefCell<TreeNode>>>> {
        if start > end {
            vec![None]
        } else {
            (start..=end).fold(Vec::new(), |mut trees, k| {
                let (left, right) = (generate_tree(start, k - 1), generate_tree(k + 1, end));
                left.into_iter().for_each(|l| {
                    right.iter().for_each(|r| {
                        let mut node = TreeNode::new(k);
                        node.left = l.clone();
                        node.right = r.clone();
                        trees.push(Some(Rc::new(RefCell::new(node))));
                    })
                });

                trees
            })
        }
    }

    if n > 0 {
        generate_tree(1, n)
    } else {
        vec![]
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "96",
    title = "Unique Binary Search Trees",
    topic = "algorithm",
    difficulty = "medium",
    tags = "Tree, BinarySearchTree, DynamicProgramming, Backtracking, Binary",
    note = "Given an integer n, return the number of structurally unique BST's (binary search trees) which has exactly n nodes of unique values from 1 to n.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/unique-binary-search-trees
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn num_trees(n: i32) -> i32 {
    // $\sum_{k=1}^{i} f(k-1) * f(i-k)$
    if n > 0 {
        let n = n as usize;
        let mut f = vec![0; n + 1];
        f[0] = 1;
        f[1] = 1;

        for i in 2..=n {
            for k in 1..=i {
                f[i] += f[k - 1] * f[i - k];
            }
        }

        f[n]
    } else {
        0
    }
}

#[inject_description(
    problems = "PROBLEMS",
    id = "97",
    title = "Interleaving String",
    topic = "algorithm",
    difficulty = "medium",
    tags = "String, DynamicProgramming",
    note = "Given strings s1, s2, and s3, find whether s3 is formed by an interleaving of s1 and s2.

An interleaving of two strings s and t is a configuration where they are divided into non-empty substrings such that:

s = s1 + s2 + ... + sn
t = t1 + t2 + ... + tm
|n - m| <= 1
The interleaving is s1 + t1 + s2 + t2 + s3 + t3 + ... or t1 + s1 + t2 + s2 + t3 + s3 + ...
Note: a + b is the concatenation of strings a and b.

Constraints:
0 <= s1.length, s2.length <= 100
0 <= s3.length <= 200
s1, s2, and s3 consist of lowercase English letters.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/interleaving-string
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn is_interleave(s1: String, s2: String, s3: String) -> bool {
    let (s1, s2, s3) = if s1.len() + s2.len() == s3.len() {
        (s1.as_bytes(), s2.as_bytes(), s3.as_bytes())
    } else {
        return false;
    };
    let mut f = vec![vec![true; s2.len() + 1]; s1.len() + 1];

    s1.iter()
        .zip(s3.iter())
        .enumerate()
        .for_each(|(i, (&x, &y))| {
            f[i + 1][0] = f[i][0] && (x == y);
        });

    s2.iter()
        .zip(s3.iter())
        .enumerate()
        .for_each(|(i, (&x, &y))| {
            f[0][i + 1] = f[0][i] && (x == y);
        });

    s1.iter().enumerate().for_each(|(i, &x)| {
        s2.iter().enumerate().for_each(|(j, &y)| {
            f[i + 1][j + 1] =
                ((x == s3[i + j + 1]) && f[i][j + 1]) || ((y == s3[i + j + 1]) && f[i + 1][j]);
        });
    });

    f[s1.len()][s2.len()]
}

#[inject_description(
    problems = "PROBLEMS",
    id = "98",
    title = "Validate Binary Search Tree",
    topic = "algorithm",
    difficulty = "medium",
    tags = "Tree, DepthFirstSearch, BinarySearchTree, BinaryTree",
    note = "Given the root of a binary tree, determine if it is a valid binary search tree (BST).

A valid BST is defined as follows:

The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/validate-binary-search-tree
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn is_valid_bst(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
    fn is_valid(
        root: Option<Rc<RefCell<TreeNode>>>,
        lower: Option<i32>,
        upper: Option<i32>,
    ) -> bool {
        match root {
            Some(x) => {
                (match (lower, upper) {
                    (Some(l), Some(u)) => x.borrow().val > l && x.borrow().val < u,
                    (Some(l), None) => x.borrow().val > l,
                    (None, Some(u)) => x.borrow().val < u,
                    _ => true,
                }) && is_valid(x.borrow().left.clone(), lower, Some(x.borrow().val))
                    && is_valid(x.borrow().right.clone(), Some(x.borrow().val), upper)
            }
            None => true,
        }
    }

    is_valid(root, None, None)
}

#[inject_description(
    problems = "PROBLEMS",
    id = "99",
    title = "Recover Binary Search Tree",
    topic = "algorithm",
    difficulty = "medium",
    tags = "Tree, DepthFirstSearch, BinarySearchTree, BinaryTree",
    note = "You are given the root of a binary search tree (BST), where the values of exactly two nodes of the tree were swapped by mistake. Recover the tree without changing its structure.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/recover-binary-search-tree
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。"
)]
pub fn recover_tree(root: &mut Option<Rc<RefCell<TreeNode>>>) {
    type TreeRoot = Option<Rc<RefCell<TreeNode>>>;
    let detect = |broken: &mut (TreeRoot, TreeRoot), prev: &mut TreeRoot, cur: &mut TreeRoot| {
        if prev.is_some()
            && prev.as_ref().unwrap().borrow().val > cur.as_ref().unwrap().borrow().val
        {
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
            while node.as_ref().unwrap().borrow().right.is_some()
                && node.as_ref().unwrap().borrow().right != cur
            {
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

    std::mem::swap(
        &mut broken.0.as_mut().unwrap().as_ref().borrow_mut().val,
        &mut broken.1.as_mut().unwrap().as_ref().borrow_mut().val,
    );
}
