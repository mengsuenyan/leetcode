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

#[cfg(test)]
mod tests {
    use crate::lc_list::ListNode;

    #[test]
    fn delete_duplicates() {
        let cases = [
            // (vec![1,1,2], vec![1,2]),
            (vec![1,1,2,3,3], vec![1,2,3]),
            // (vec![], vec![]),
        ];
        
        for c in cases.iter() {
            let l = ListNode::from_vec(&c.0);
            assert_eq!(super::delete_duplicates(l), ListNode::from_vec(&c.1));
        }
    }
}
