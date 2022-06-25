use leetcode_rs::easy::p1::{ListNode, TreeNode};
use leetcode_rs::easy::p4;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

#[test]
fn test_summary_ranges() {
    let cases = vec![
        (vec![], vec![]),
        (
            vec![0, 1, 2, 4, 5, 7],
            vec![format!("0->2"), format!("4->5"), format!("7")],
        ),
        (
            vec![0, 2, 3, 4, 6, 8, 9],
            vec![format!("0"), format!("2->4"), format!("6"), format!("8->9")],
        ),
        (vec![1], vec![format!("1")]),
    ];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        assert_eq!(p4::summary_ranges(in1), out1, "case {} failed", i);
    }
}

#[test]
fn test_is_power_of_two() {
    let cases = vec![(1, true), (16, true), (3, false), (4, true), (5, false)];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        assert_eq!(p4::is_power_of_two(in1), out1, "case {} failed", i);
    }
}

#[test]
fn test_is_palindrome() {
    let cases = vec![
        (vec![1, 2, 2, 1], true),
        (vec![1, 2], false),
        (vec![], true),
        (vec![1], true),
    ];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        assert_eq!(
            p4::is_palindrome(ListNode::from_slice(in1.as_slice())),
            out1,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_lowest_common_ancestor() {
    let cases = vec![
        (vec![6, 2, 8, 0, 4, 7, 9, i32::MAX, i32::MAX, 3, 5], 2, 8, 6),
        (vec![6, 2, 8, 0, 4, 7, 9, i32::MAX, i32::MAX, 3, 5], 2, 4, 2),
        (vec![2, 1], 2, 1, 2),
    ];

    for (i, (in1, p, q, out1)) in cases.into_iter().enumerate() {
        let (root, p, q) = (
            TreeNode::from_slice(in1.as_slice()),
            Some(Rc::new(RefCell::new(TreeNode::new(p)))),
            Some(Rc::new(RefCell::new(TreeNode::new(q)))),
        );
        assert_eq!(
            p4::lowest_common_ancestor(root, p, q).unwrap().borrow().val,
            out1,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_is_anagram() {
    let cases = vec![
        ("anagram", "nagaram", true),
        ("rat", "car", false),
        ("fff", "ff", false),
    ];

    for (i, (in1, in2, out1)) in cases.into_iter().enumerate() {
        assert_eq!(
            p4::is_anagram(in1.to_string(), in2.to_string()),
            out1,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_binary_tree_paths() {
    let cases = vec![
        (vec![1, 2, 3, i32::MAX, 5], vec!["1->2->5", "1->3"]),
        (vec![1], vec!["1"]),
        (vec![], vec![]),
    ];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        let o = p4::binary_tree_paths(TreeNode::from_slice(in1.as_slice()));
        assert_eq!(o.len(), out1.len(), "case {} failed", i);
        let h = o.into_iter().collect::<HashSet<_>>();
        for e in out1 {
            assert!(h.contains(e), "case {} failed", i);
        }
    }
}

#[test]
fn test_add_digits() {
    let cases = vec![(38, 2), (0, 0)];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        assert_eq!(p4::add_digits(in1), out1, "case {} failed", i);
    }
}

#[test]
fn test_is_ugly() {
    let cases = vec![(6, true), (1, true), (14, false)];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        assert_eq!(p4::is_ugly(in1), out1, "case {} failed", i);
    }
}

#[test]
fn test_missing_number() {
    let cases = vec![
        (vec![3, 0, 1], 2),
        (vec![0, 1], 2),
        (vec![9, 6, 4, 2, 3, 5, 7, 0, 1], 8),
    ];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        assert_eq!(p4::missing_number(in1), out1, "case {} failed", i);
    }
}

#[test]
fn test_move_zeroes() {
    let cases = vec![
        (vec![0, 1, 0, 3, 12], vec![1, 3, 12, 0, 0]),
        (vec![0], vec![0]),
    ];

    for (i, (mut in1, out1)) in cases.into_iter().enumerate() {
        p4::move_zeroes(&mut in1);
        assert_eq!(in1, out1, "case {} failed", i);
    }
}

#[test]
fn test_word_pattern() {
    let cases = vec![
        ("abba", "dog cat cat dog", true),
        ("abba", "dog cat cat fish", false),
        ("aaaa", "dog cat cat dog", false),
    ];

    for (i, (in1, in2, out1)) in cases.into_iter().enumerate() {
        assert_eq!(
            p4::word_pattern(in1.to_string(), in2.to_string()),
            out1,
            "case {} failed",
            i
        );
    }
}
