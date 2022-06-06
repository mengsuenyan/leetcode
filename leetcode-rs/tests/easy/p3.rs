use leetcode_rs::easy::p1::{ListNode, TreeNode};
use leetcode_rs::easy::p3;

#[test]
fn test_title_to_number() {
    let cases = vec![("A", 1), ("AB", 28), ("ZY", 701), ("FXSHRXW", 2147483647)];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p3::title_to_number(input.to_string()),
            output,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_is_happy() {
    let cases = vec![(19, true), (2, false)];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(p3::is_happy(input), output, "case {} failed", i);
    }
}

#[test]
fn test_remove_elements() {
    let cases = vec![
        (vec![1, 2, 6, 3, 4, 5, 6], 6, vec![1, 2, 3, 4, 5]),
        (vec![], 1, vec![]),
        (vec![7, 7, 7, 7], 7, vec![]),
    ];

    for (i, (input, target, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p3::remove_elements(ListNode::from_slice(input.as_slice()), target),
            ListNode::from_slice(output.as_slice()),
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_is_isomorphic() {
    let cases = vec![
        ("egg", "add", true),
        ("", "", true),
        ("foo", "bar", false),
        ("paper", "title", true),
    ];

    for (i, (in1, in2, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p3::is_isomorphic(in1.to_string(), in2.to_string()),
            output,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_reverse_list() {
    let cases = vec![
        (vec![], vec![]),
        (vec![1, 2, 3, 4, 5], vec![5, 4, 3, 2, 1]),
        (vec![1, 2], vec![2, 1]),
    ];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p3::reverse_list(ListNode::from_slice(input.as_slice())),
            ListNode::from_slice(output.as_slice()),
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_contains_duplicate() {
    let cases = vec![
        (vec![1, 2, 3, 1], true),
        (vec![1, 2, 3, 4], false),
        (vec![1, 1, 1, 3, 3, 4, 3, 2, 4, 2], true),
    ];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(p3::contains_duplicate(input), output, "case {} failed", i);
    }
}

#[test]
fn test_contains_nearby_duplicate() {
    let cases = vec![
        (vec![1, 2, 3, 1], 3, true),
        (vec![1, 0, 1, 1], 1, true),
        (vec![1, 2, 3, 1, 2, 3], 2, false),
    ];

    for (i, (input, k, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p3::contains_nearby_duplicate(input, k),
            output,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_invert_tree() {
    let cases = vec![
        (vec![4, 2, 7, 1, 3, 6, 9], vec![4, 7, 2, 9, 6, 3, 1]),
        (vec![2, 1, 3], vec![2, 3, 1]),
        (vec![], vec![]),
    ];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            TreeNode::to_vec(p3::invert_tree(TreeNode::from_slice(input.as_slice()))),
            output,
            "case {} failed",
            i
        );
    }
}
