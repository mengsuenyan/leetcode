//! 链表相关

use std::boxed::Box;

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
  pub val: i32,
  pub next: Option<Box<ListNode>>
}

impl ListNode {
    #[allow(dead_code)]
    #[inline]
    fn new(val: i32) -> Self {
        ListNode {
            next: None,
            val
        }
    }
}

impl ListNode {
    fn from_vec(v: &Vec<i32>) -> Option<Box<ListNode>> {
        if v.is_empty() {
            return None;
        }
        
        let mut itr = v.iter().rev();
        let mut h = Some(Box::new(ListNode::new(*itr.next().unwrap())));
        for &ele in itr {
            h = Some(Box::new(ListNode {
                val: ele,
                next: h,
            }));
        }
        
        h
    }
    
    /// TODO: 接收Option<Box<ListNode>>的函数, 再调用to_vec, 然后处理数组后, 再from_vec转为list, 会造成内存泄漏
    fn to_vec(head: &Option<Box<ListNode>>) -> Vec<i32>
    {
        let mut v = Vec::new();
        let mut head = head;
        
        while head.is_some() {
            v.push(head.as_ref().unwrap().val);
            head = &head.as_ref().unwrap().next;
        }
        
        v
    }
    
    fn len(mut head: &Option<Box<ListNode>>) -> usize {
        let mut l = 0;
        while let Some(x) = head {
            l += 1;
            head = &x.next;
        }
        l
    }
}

/// Given a sorted linked list, delete all duplicates such that each element appear only once.  
///   
/// 来源：力扣（LeetCode）  
/// 链接：https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list  
/// 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。  
/// O( n )
pub fn delete_duplicates(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut head = head;
    let mut h = &mut head;
    
    while h.is_some() && h.as_ref().unwrap().next.is_some() {
        if h.as_ref().unwrap().val == h.as_ref().unwrap().next.as_ref().unwrap().val {
            h.as_mut().unwrap().next = h.as_mut().unwrap().next.as_mut().unwrap().next.take();
        } else {
            h = &mut h.as_mut().unwrap().next;
        }
    }
    
    head
}

/// Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists.  
///   
/// 来源：力扣（LeetCode）  
/// 链接：https://leetcode-cn.com/problems/merge-two-sorted-lists  
/// 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。  
/// O(n)
pub fn merge_two_lists(l1: Option<Box<ListNode>>, l2: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    if l1.is_none() {
        return l2;
    } else if l2.is_none() {
        return l1;
    }
    
    let (mut l1, mut l2) = (l1, l2);
    let (mut ml1, mut ml2) = (&mut l1, &mut l2);
    let mut head: Option<Box<ListNode>> = None;
    let mut h= &mut head;
    
    loop {
        let val = if ml1.is_some() && ml2.is_some() {
            if ml1.as_ref().unwrap().val < ml2.as_ref().unwrap().val {
                let tmp = ml1.as_ref().unwrap().val;
                ml1 = &mut ml1.as_mut().unwrap().next;
                tmp
            } else {
                let tmp = ml2.as_ref().unwrap().val;
                ml2 = &mut ml2.as_mut().unwrap().next;
                tmp
            }
            
        } else if ml1.is_some() {
            let tmp = ml1.as_ref().unwrap().val;
            ml1 = &mut ml1.as_mut().unwrap().next;
            tmp
        } else if ml2.is_some() {
            let tmp = ml2.as_ref().unwrap().val;
            ml2 = &mut ml2.as_mut().unwrap().next;
            tmp
        } else {
            break;
        };
        
        if h.is_some() {
            h.as_mut().unwrap().next = Some(Box::new(ListNode::new(val)));
            h = &mut h.as_mut().unwrap().next;
        } else {
            head = Some(Box::new(ListNode::new(val)));
            h = &mut head;
        }
    }
    
    head
}

/// Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.  
///   
/// You should preserve the original relative order of the nodes in each of the two partitions.  
///   
/// 来源：力扣（LeetCode）  
/// 链接：https://leetcode-cn.com/problems/partition-list  
/// 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。  
/// O(n)
pub fn partition(head: Option<Box<ListNode>>, x: i32) -> Option<Box<ListNode>> {
    if head.is_none() {
        return head;
    }
    
    let mut head = head;
    let mut h = &mut head;
    let mut lt = None;
    let mut gt =None;
    let mut ltr = &mut lt;
    let mut gtr = &mut gt;
    
    loop {
        let val = if h.is_some() {
            let tmp = h.as_ref().unwrap().val;
            h = &mut h.as_mut().unwrap().next;
            tmp
        } else {
            break;
        };
        
        if val < x {
            if ltr.is_none() {
                lt = Some(Box::new(ListNode::new(val)));
                ltr = &mut lt;
            } else {
                ltr.as_mut().unwrap().next = Some(Box::new(ListNode::new(val)));
                ltr = &mut ltr.as_mut().unwrap().next;
            }
        } else {
            if gtr.is_none() {
                gt = Some(Box::new(ListNode::new(val)));
                gtr = &mut gt;
            } else {
                gtr.as_mut().unwrap().next = Some(Box::new(ListNode::new(val)));
                gtr = &mut gtr.as_mut().unwrap().next;
            }
        }
    }
    
    if ltr.is_none() {
        gt
    } else {
        ltr.as_mut().unwrap().next = gt;
        lt
    }
    
}

/// Reverse a linked list from position m to n. Do it in one-pass.  
///   
/// Note: 1 ≤ m ≤ n ≤ length of list.  
///   
/// 来源：力扣（LeetCode）  
/// 链接：https://leetcode-cn.com/problems/reverse-linked-list-ii  
/// 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。  
/// O(n) 
pub fn reverse_between(head: Option<Box<ListNode>>, m: i32, n: i32) -> Option<Box<ListNode>> {
    if head.is_none() || m == n {
        return head;
    }
    
    let mut h = &head;
    let mut re = Vec::new();
    
    while h.is_some() {
        re.push(h.as_ref().unwrap().val);
        h = &h.as_ref().unwrap().next;
    }

    let (m, n) = ((m-1) as usize, (n-1) as usize);
    let sub = &mut (re.as_mut_slice())[m..=n];
    sub.reverse();
    
    ListNode::from_vec(&re)
}

/// Given a linked list, reverse the nodes of a linked list k at a time and return its modified list.  
///   
/// k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k then left-out nodes in the end should remain as it is.  
///   
/// 来源：力扣（LeetCode）  
/// 链接：https://leetcode-cn.com/problems/reverse-nodes-in-k-group  
/// 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。  
/// O(n)
pub fn reverse_k_group(head: Option<Box<ListNode>>, k: i32) -> Option<Box<ListNode>> {
    let mut h = &head;
    let mut re = Vec::new();
    
    while h.is_some() {
        re.push(h.as_ref().unwrap().val);
        h = &h.as_ref().unwrap().next;
    }
    
    let k = k as usize;
    let num = re.len() / k;
    let res = re.as_mut_slice();
    for i in 0..num
    {
        let (start, end) = (i * k, i * k + k);
        let sub = &mut res[start..end];
        sub.reverse();
    }
    
    ListNode::from_vec(&re)
}

/// Given a linked list, remove the n-th node from the end of list and return its head.  
///   
/// 来源：力扣（LeetCode）  
/// 链接：https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list  
/// 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。  
/// O(n)
pub fn remove_nth_from_end(head: Option<Box<ListNode>>, n: i32) -> Option<Box<ListNode>> {  
    let mut v = ListNode::to_vec(&head);
    
    v.remove(v.len() - (n as usize));
    
    ListNode::from_vec(&v)
}

/// Add Two Numbers
///
/// You are given two linked lists representing two non-negative numbers. e digits are stored in reverse
/// order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.
/// Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
/// Output: 7 -> 0 -> 8
pub fn add_two_numbers(l1: Option<Box<ListNode>>, l2: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let (mut l1, mut l2, mut carry) = (l1.as_ref(), l2.as_ref(), 0);
    let mut head = Box::new(ListNode::new(0));
    let mut tail = head.as_mut();
    
    loop {
        let val = match (l1, l2) {
            (Some(x), Some(y)) => {
                l1 = x.next.as_ref();
                l2 = y.next.as_ref();
                x.val + y.val + carry
            },
            (Some(x), None) => {
                l1 = x.next.as_ref();
                x.val + carry
            },
            (None, Some(y)) => {
                l2 = y.next.as_ref();
                y.val + carry
            }
            _ => break,
        };
        
        let val = if val > 9 {carry = 1; val - 10} else {carry = 0; val};
        tail.next = Some(Box::new(ListNode::new(val)));
        tail = tail.next.as_mut().unwrap();
    }

    if carry > 0 {
        tail.next = Some(Box::new(ListNode::new(carry)));
    }
    head.next
}

/// Remove Duplicates from Sorted List II
/// 
/// Given a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers
/// from the original list.
/// For example,
/// Given 1->2->3->3->4->4->5, return 1->2->5.
/// Given 1->1->1->2->3, return 2->3
pub fn delete_duplicates_ii(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    if head.is_none() {
        return None;
    }
    
    let (mut prev, mut tail) = (head.as_ref().unwrap().val, 
                                &head.as_ref().unwrap().next);
    let (mut v, mut is_push) = (vec![prev], true);

    while let Some(x) = tail.as_ref() {
        let tmp = x.val;
        if tmp != prev {
            prev = tmp;
            is_push = true;
            v.push(tmp);
        } else {
            if is_push {
                v.pop();
                is_push = false;
            }
        }
        tail = &x.next;
    }
    

    ListNode::from_vec(&v)
}

/// Rotate List
/// 
/// Given a list, rotate the list to the right by k places, where k is non-negative.
/// For example: Given 1->2->3->4->5->nullptr and k = 2, return 4->5->1->2->3->nullptr.
pub fn rotate_right(head: Option<Box<ListNode>>, k :i32) -> Option<Box<ListNode>> {
    if head.is_some() {
        let mut v = ListNode::to_vec(&head);
        let k = (k as usize) % v.len();

        v.rotate_right(k);
        ListNode::from_vec(&v)
    } else {
        None
    }
}

/// Swap Nodes in Pairs
/// 
/// Given a linked list, swap every two adjacent nodes and return its head.
/// For example, Given 1->2->3->4, you should return the list as 2->1->4->3.
/// Your algorithm should use only constant space. You may not modify the values in the list, only nodes
/// itself can be changed
pub fn swap_pairs(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut head = head;
    let mut anchor = ListNode::new(0);
    let mut tail = &mut anchor;
    
    while let Some(mut x) = head {
        head = x.next.take();
        if let Some(mut y) = head {
            head = y.next.take();
            y.next = Some(x);
            tail.next = Some(y);
            tail = tail.next.as_mut().unwrap().next.as_mut().unwrap();
        } else {
            tail.next = Some(x);
            tail = tail.next.as_mut().unwrap();
        }
    }
    
    anchor.next
}

/// Linked List Cycle
/// 
/// Given a linked list, determine if it has a cycle in it.
/// Follow up: Can you solve it without using extra space?
pub fn has_cycle(head: Option<Box<ListNode>>) -> bool {
    let (mut slow, mut fast) = (head.as_ref(), head.as_ref());
    while fast.is_some() && fast.unwrap().next.is_some() {
        slow = slow.unwrap().next.as_ref();
        fast = fast.unwrap().next.as_ref().unwrap().next.as_ref();
        if slow == fast {
            return true;
        }
    }
    false
}

/// Linked List Cycle II
/// 
/// Given a linked list, return the node where the cycle begins. If there is no cycle, return null.
/// Follow up: Can you solve it without using extra space?
pub fn detect_cycle(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let (mut slow, mut fast) = (head.as_ref(), head.as_ref());
    
    while fast.is_some() && fast.unwrap().next.is_some() {
        slow = slow.unwrap().next.as_ref();
        fast = fast.unwrap().next.as_ref().unwrap().next.as_ref();
        if slow == fast {
            // slow: 相遇点记为x, 假设和slow2可以相遇则走了 (r-x) + n*r;
            // 此假设成立的条件是slow2走了(r-x) + n*r后在x点;
            // 记环起点为y, (r-x)+n*r=(L-y-x)+n*r正是x点位置, 故成立;
            let mut slow2 = head.as_ref();
            while slow != slow2 {
                slow = slow.unwrap().next.as_ref();
                slow2 = slow2.unwrap().next.as_ref();
            }
            return Some(Box::new(ListNode::new(slow2.unwrap().val)));
        }
    }
    None
}

/// Reorder List
/// 
/// Given a singly linked list L : L0 -> L1 -> · · · -> Ln−1 -> Ln, reorder it to: L0 -> Ln -> L1 ->
/// Ln−1 -> L2 -> Ln−2 -> · · ·
/// You must do this in-place without altering the nodes’ values.
/// For example, Given {1,2,3,4}, reorder it to {1,4,2,3}
pub fn reorder_list(head: &mut Option<Box<ListNode>>) {
    let mut headc = head.clone();
    let len = ListNode::len(&headc);
    let (mut half1, mut half2) = (ListNode::new(0), ListNode::new(0));
    let (mut tail, mut cnt) = (&mut half1, 0);
    let mut half3 = None;
    
    // split
    let pivot = len >> 1;
    while let Some(mut x) = headc {
        headc = x.next.take();
        if cnt < pivot {
            tail.next = Some(x);
            tail = tail.next.as_mut().unwrap();
        } else if cnt > pivot {
            x.next = half2.next.take();
            half2.next = Some(x);
        } else if (cnt == pivot) && (len & 0x1 == 1) {
            half3 = Some(x);
        } else {
            x.next = half2.next.take();
            half2.next = Some(x);
        }
        cnt += 1;
    }
    
    // merge
    let mut res = ListNode::new(0);
    tail = &mut res;
    let (mut half1, mut half2) = (half1.next, half2.next);
    while let (Some(mut x), Some(mut y)) = (half1, half2) {
        half1 = x.next.take();
        half2 = y.next.take();
        tail.next = Some(x);
        tail.next.as_mut().unwrap().next = Some(y);
        tail = tail.next.as_mut().unwrap().next.as_mut().unwrap();
    }

    tail.next = half3;
    
    *head = res.next;
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * let obj = LRUCache::new(capacity);
 * let ret_1: i32 = obj.get(key);
 * obj.put(key, value);
 */
/// LRU cache
/// 
/// Design and implement a data structure for Least Recently Used (LRU) cache. It should support the
/// following operations: get and set.
/// get(key) - Get the value (will always be positive) of the key if the key exists in the cache, otherwise
/// return -1.
/// set(key, value) - Set or insert the value if the key is not already present. When the cache reached
/// its capacity, it should invalidate the least recently used item before inserting a new item
#[allow(unused)]
struct LRUCache {
    // key, (value, slotId)
    cache: std::collections::HashMap<i32, (i32, usize)>,
    // slotId, key
    slot: std::collections::HashMap<usize, i32>,
    cur_time: usize,
    old_time: usize,
    cap: usize,
}

/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl LRUCache {

    #[allow(unused)]
    fn new(capacity: i32) -> Self {
        let capacity = capacity as usize;
        LRUCache {
            cache: std::collections::HashMap::with_capacity(capacity),
            slot: std::collections::HashMap::with_capacity(capacity),
            cur_time: 0,
            old_time: 0,
            cap: capacity,
        }
    }

    fn update(&mut self, key: i32, time: usize) {
        self.slot.remove(&time);
        self.slot.insert(self.cur_time, key);
        if !self.slot.contains_key(&self.old_time) {
            self.old_time += 1;
        }
    }

    #[allow(unused)]
    fn get(&mut self, key: i32) -> i32 {
        match self.cache.get(&key) {
            Some(&x) => {
                self.cur_time += 1;
                self.cache.insert(key, (x.0, self.cur_time));
                
                self.update(key, x.1);
                x.0
            },
            None => {-1},
        }
    }

    #[allow(unused)]
    fn put(&mut self, key: i32, value: i32) {
        self.cur_time += 1;
        
        // 时间更新
        match self.cache.get(&key) {
            Some(&x) => {
                self.update(key, x.1);
            },
            None => {
                if self.cache.len() < self.cap {
                    self.slot.insert(self.cur_time, key);
                } else {
                    'out: loop {
                        while let Some(&x) = self.slot.get(&self.old_time) {
                            self.slot.remove(&self.old_time);
                            self.slot.insert(self.cur_time, key);
                            self.cache.remove(&x);
                            self.old_time += 1;
                            break 'out;
                        }
                        self.old_time += 1;
                    }
                }
            }
        }

        self.cache.insert(key, (value, self.cur_time));
    }
}

/// Merge k Sorted Lists
/// 
/// Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity
pub fn merge_k_lists(lists: Vec<Option<Box<ListNode>>>) -> Option<Box<ListNode>> {
    // ts: O(n), sc: O(n)
    let mut bh = std::collections::BinaryHeap::with_capacity(lists.len() << 6);
    
    lists.iter().for_each(|mut x| {
        while let Some(ele) = x {
            bh.push(ele.val);
            x = &ele.next;
        }
    });
    
    ListNode::from_vec(&bh.into_sorted_vec())
}

/// Insertion sort list
/// 
/// Sort a linked list using insertion sort
pub fn insertion_sort_list(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut anchor = ListNode::new(0);
    let mut head = head;
    
    while let Some(mut node) = head {
        head = node.next.take();
        
        let mut tail = &mut anchor;
        while tail.next.is_some() && (tail.next.as_deref_mut()?.val < node.val) {
            tail = tail.next.as_deref_mut()?;
        }
        
        node.next = tail.next.take();
        tail.next = Some(node);
    }
    
    anchor.next
}

/// Sort List
/// 
/// Sort a linked list in O(nlogn) time using constant space complexity
pub fn sort_list(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let len = |mut x: &Option<Box<ListNode>>|{
        let mut cnt = 0;
        while let Some(node) = x {
            cnt += 1;
            x = &node.next;
        }
        cnt
    };
    
    let len = len(&head);
    sort_list_help1_(head, len)
}

/// split list
fn sort_list_help1_(mut list: Option<Box<ListNode>>, len: usize) -> Option<Box<ListNode>> {
    if list.is_none() || list.as_deref()?.next.is_none() {
        return list;
    }
    
    let half_palce = (len + 1) >> 1;
    let mut right = &mut list;
    for _ in 0..half_palce {
        right = &mut right.as_deref_mut()?.next;
    }
    
    let right = right.take();
    let left = sort_list_help1_(list, half_palce);
    let right = sort_list_help1_(right, len - half_palce);
    
    sort_list_help2_(left, right)
}

/// merge list
fn sort_list_help2_(mut left: Option<Box<ListNode>>, mut right: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut anchor = ListNode::new(0);
    let mut tail = &mut anchor.next;
    
    'out: loop {
        
        let is_left = if left.is_some() && right.is_some() {
            if left.as_deref()?.val < right.as_deref()?.val { true } else { false }
        } else if left.is_some() { true
        } else if right.is_some() { false
        } else { break 'out; };
        
        *tail = if is_left {
            let mut tmp= left.take();
            left = tmp.as_deref_mut()?.next.take();
            tmp
        } else {
            let mut tmp = right.take();
            right = tmp.as_deref_mut()?.next.take();
            tmp
        };
        
        tail = &mut tail.as_deref_mut()?.next;
    }
    
    anchor.next
}


#[cfg(test)]
mod tests {
    use crate::lc_list::{ListNode, LRUCache};

    #[test]
    fn delete_duplicates() {
        let cases = [
            (vec![1,1,2], vec![1,2]),
            (vec![1,1,2,3,3], vec![1,2,3]),
            (vec![], vec![]),
        ];
        
        for c in cases.iter() {
            let l = ListNode::from_vec(&c.0);
            assert_eq!(super::delete_duplicates(l), ListNode::from_vec(&c.1));
        }
    }
    
    #[test]
    fn merge_two_lists() {
        let cases = [
            (vec![1,2,4], vec![1,3,4], vec![1,1,2,3,4,4]),
            (vec![1,2,4], vec![], vec![1,2,4]),
            (vec![], vec![1,3,4], vec![1,3,4]),
            (vec![], vec![], vec![]),
        ];

        for c in cases.iter() {
            let l1 = ListNode::from_vec(&c.0);
            let l2 = ListNode::from_vec(&c.1);
            assert_eq!(super::merge_two_lists(l1, l2), ListNode::from_vec(&c.2));
        }
    }
    
    #[test]
    fn partition() {
        let cases = [
            (vec![1,4,3,2,5,2], 3, vec![1,2,2,4,3,5]),
        ];

        for c in cases.iter() {
            let l1 = ListNode::from_vec(&c.0);
            assert_eq!(super::partition(l1, c.1), ListNode::from_vec(&c.2));
        }
    }
    
    #[test]
    fn add_two_numbers() {
        let cases = [
            ((vec![2,4,3], vec![5,6,4]), vec![7,0,8]),
            ((vec![9],vec![9]), vec![8,1]),
        ];
        cases.iter().for_each(|x| {
            let tmp = super::add_two_numbers(ListNode::from_vec(&(x.0).0),
                                             ListNode::from_vec(&(x.0).1));
            assert_eq!(x.1, ListNode::to_vec(&tmp), "case: {:?}", x.0);
        })
    }
    
    #[test]    
    fn delete_duplicates_ii() {
        let cases = [
            (vec![1,2,3,3,4,4,5], vec![1,2,5]),
            (vec![1,1,1,2,3], vec![2,3]),
        ];
        
        cases.iter().for_each(|x| {
            let tmp = super::delete_duplicates_ii(ListNode::from_vec(&x.0));
            assert_eq!(x.1, ListNode::to_vec(&tmp), "case: {:?}", x.0);
        })
    }
    
    #[test]
    fn rotate_right() {

        let cases = [
            ((vec![1,2,3,4,5], 2), vec![4,5,1,2,3]),
            ((vec![0,1,2],4), vec![2,0,1]),
        ];

        cases.iter().for_each(|x| {
            let tmp = super::rotate_right(ListNode::from_vec(&(x.0).0), (x.0).1);
            assert_eq!(x.1, ListNode::to_vec(&tmp), "case: {:?}", x.0);
        })
    }
    
    #[test]
    fn swap_pairs() {

        let cases = [
            (vec![1,2,3,4], vec![2,1,4,3]),
        ];

        cases.iter().for_each(|x| {
            let tmp = super::swap_pairs(ListNode::from_vec(&x.0));
            assert_eq!(x.1, ListNode::to_vec(&tmp), "case: {:?}", x.0);
        })
    }
    
    #[test]
    fn reorder_list() {

        let cases = [
            (vec![1,2,3,4], vec![1,4,2,3]),
            (vec![1,2,3,4,5], vec![1,5,2,4,3]),
        ];

        cases.iter().for_each(|x| {
            let mut tmp = &mut ListNode::from_vec(&x.0);
            super::reorder_list(&mut tmp);
            assert_eq!(x.1, ListNode::to_vec(&tmp), "case: {:?}", x.0);
        })
    }
    
    #[test]
    fn lru_cache() {
        let mut cache = LRUCache::new(2);
        cache.put(1, 1);
        cache.put(2, 2);
        assert_eq!(1, cache.get(1));       // 返回  1
        cache.put(3, 3);    // 该操作会使得关键字 2 作废
        assert_eq!(-1, cache.get(2));       // 返回 -1 (未找到)
        cache.put(4, 4);    // 该操作会使得关键字 1 作废
        assert_eq!(-1, cache.get(1));       // 返回 -1 (未找到)
        assert_eq!(3, cache.get(3));       // 返回  3
        assert_eq!(4, cache.get(4));       // 返回  4
    }
    
    #[test]
    fn insertion_sort_list() {
        let cases = [
            (vec![], vec![]),
            (vec![4,2,1,3], vec![1,2,3,4]),
            (vec![-1,5,3,4,0], vec![-1,0,3,4,5]),
        ];
        
        cases.iter().for_each(|x| {
            let tmp = super::insertion_sort_list(ListNode::from_vec(&x.0));
            let tmp = ListNode::to_vec(&tmp);
            assert_eq!(x.1, tmp, "cases: {:?}", x.0);
        });
    }
    
    #[test]
    fn sort_list() {
        let cases = [
            (vec![], vec![]),
            (vec![4,2,1,3], vec![1,2,3,4]),
            (vec![-1,5,3,4,0], vec![-1,0,3,4,5]),
        ];

        cases.iter().for_each(|x| {
            let tmp = super::sort_list(ListNode::from_vec(&x.0));
            let tmp = ListNode::to_vec(&tmp);
            assert_eq!(x.1, tmp, "cases: {:?}", x.0);
        });
    }
}
