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


    #[allow(dead_code)]
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

#[cfg(test)]
mod tests {
    use crate::lc_list::ListNode;

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
}
