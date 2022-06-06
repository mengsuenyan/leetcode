use leetcode_rs::easy::p1::TreeNode;
use leetcode_rs::easy::p2;

#[test]
fn test_is_symmetric() {
    let cases = vec![
        (vec![1, 2, 2, 3, 4, 4, 3], true),
        (vec![1, 2, 2, i32::MAX, 3, i32::MAX, 3], false),
        (vec![], true),
        (vec![1], true),
    ];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p2::is_symmetric(TreeNode::from_slice(input.as_slice())),
            output,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_max_depth() {
    let cases = vec![
        (vec![3, 9, 20, i32::MAX, i32::MAX, 15, 7], 3),
        (vec![1, i32::MAX, 2], 2),
    ];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p2::max_depth(TreeNode::from_slice(input.as_slice())),
            output,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_sorted_array_to_bst() {
    let cases = vec![
        (vec![-10, -3, 0, 5, 9], vec![0, -3, 9, -10, i32::MAX, 5]),
        (vec![1, 3], vec![3, 1]),
        (vec![], vec![]),
    ];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        let res = p2::sorted_array_to_bst(input);
        // println!("{:?}", TreeNode::to_vec(res));
        assert_eq!(TreeNode::to_vec(res), output, "case {} failed", i);
    }
}

#[test]
fn test_is_balanced() {
    let cases = vec![
        (vec![3, 9, 20, i32::MAX, i32::MAX, 15, 7], true),
        (vec![1, 2, 2, 3, 3, i32::MAX, i32::MAX, 4, 4], false),
        (vec![], true),
        (vec![1], true),
        (vec![1, 2], true),
    ];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p2::is_balanced(TreeNode::from_slice(input.as_slice())),
            output,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_min_depth() {
    let cases = vec![
        (vec![3, 9, 20, i32::MAX, i32::MAX, 15, 7], 2),
        (
            vec![2, i32::MAX, 3, i32::MAX, 4, i32::MAX, 5, i32::MAX, 6],
            5,
        ),
        (vec![], 0),
        (vec![1], 1),
    ];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p2::min_depth(TreeNode::from_slice(input.as_slice())),
            output,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_has_path_sum() {
    let cases = vec![
        (
            vec![
                5,
                4,
                8,
                11,
                i32::MAX,
                13,
                4,
                7,
                2,
                i32::MAX,
                i32::MAX,
                i32::MAX,
                1,
            ],
            22,
            true,
        ),
        (vec![1, 2, 3], 5, false),
        (vec![], 0, false),
        (vec![1], 1, true),
    ];
    for (i, (input, target, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p2::has_path_sum(TreeNode::from_slice(input.as_slice()), target),
            output,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_generate() {
    let cases = vec![
        (
            5,
            vec![
                vec![1],
                vec![1, 1],
                vec![1, 2, 1],
                vec![1, 3, 3, 1],
                vec![1, 4, 6, 4, 1],
            ],
        ),
        (1, vec![vec![1]]),
    ];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(p2::generate(input), output, "case {} failed", i);
    }
}

#[test]
fn test_get_row() {
    let cases = vec![(3, vec![1, 3, 3, 1]), (0, vec![1])];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(p2::get_row(input), output, "case {} failed", i);
    }
}

#[test]
fn test_max_profit() {
    let cases = vec![
        (vec![7, 1, 5, 3, 6, 4], 5),
        (vec![7, 6, 4, 3, 1], 0),
        (vec![], 0),
        (vec![1], 0),
    ];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(p2::max_profit(input), output, "case {} failed", i);
    }
}

#[test]
fn test_is_palindrome() {
    let cases = vec![
        (" ", true),
        (",", true),
        (", ", true),
        ("race a car", false),
        ("A man, a plan, a canal: Panama", true),
    ];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p2::is_palindrome(input.to_string()),
            output,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_single_number() {
    let cases = vec![(vec![2, 2, 1], 1), (vec![1], 1), (vec![4, 1, 2, 1, 2], 4)];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(p2::single_number(input), output, "case {} failed", i);
    }
}

#[test]
fn test_preorder_traversal() {
    let cases = vec![
        (vec![1, i32::MAX, 2, 3], vec![1, 2, 3]),
        (vec![], vec![]),
        (vec![1], vec![1]),
    ];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p2::preorder_traversal(TreeNode::from_slice(input.as_slice())),
            output,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_postorder_traversal() {
    let cases = vec![
        (vec![1, i32::MAX, 2, 3], vec![3, 2, 1]),
        (vec![], vec![]),
        (vec![1], vec![1]),
    ];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p2::postorder_traversal(TreeNode::from_slice(input.as_slice())),
            output,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_convert_to_title() {
    let cases = vec![(1, "A"), (28, "AB"), (701, "ZY")];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(p2::convert_to_title(input), output, "case {} failed", i);
    }
}

#[test]
fn test_majority_element() {
    let cases = vec![(vec![3, 2, 3], 3), (vec![2, 2, 1, 1, 1, 2, 2], 2)];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(p2::majority_element(input), output, "case {} failed", i);
    }
}
