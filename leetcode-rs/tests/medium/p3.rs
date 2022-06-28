use leetcode_rs::easy::p1::{ListNode, TreeNode};
use leetcode_rs::medium::p3;
use std::collections::HashSet;

#[test]
fn test_set_zeroes() {
    let cases = vec![
        (
            vec![vec![1, 1, 1], vec![1, 0, 1], vec![1, 1, 1]],
            vec![vec![1, 0, 1], vec![0, 0, 0], vec![1, 0, 1]],
        ),
        (
            vec![vec![0, 1, 2, 0], vec![3, 4, 5, 2], vec![1, 3, 1, 5]],
            vec![vec![0, 0, 0, 0], vec![0, 4, 5, 0], vec![0, 3, 1, 0]],
        ),
        (vec![vec![]], vec![vec![]]),
    ];

    for (i, (mut in1, out1)) in cases.into_iter().enumerate() {
        p3::set_zeroes(&mut in1);
        assert_eq!(in1, out1, "case {} failed", i);
    }
}

#[test]
fn test_search_matrix() {
    let cases = vec![
        (
            vec![vec![1, 3, 5, 7], vec![10, 11, 16, 20], vec![23, 30, 34, 60]],
            3,
            true,
        ),
        (
            vec![vec![1, 3, 5, 7], vec![10, 11, 16, 20], vec![23, 30, 34, 60]],
            13,
            false,
        ),
        (vec![vec![]], 1, false),
    ];

    for (i, (in1, in2, out1)) in cases.into_iter().enumerate() {
        assert_eq!(p3::search_matrix(in1, in2), out1, "case {} failed", i);
    }
}

#[test]
fn test_sort_colors() {
    let cases = vec![
        (vec![2, 0, 2, 1, 1, 0], vec![0, 0, 1, 1, 2, 2]),
        (vec![2, 0, 1], vec![0, 1, 2]),
    ];

    for (i, (mut in1, out1)) in cases.into_iter().enumerate() {
        p3::sort_colors(&mut in1);
        assert_eq!(in1, out1, "case {} failed", i);
    }
}

#[test]
fn test_combine() {
    let cases = vec![
        (1, 1, vec![vec![1]]),
        (
            4,
            2,
            vec![
                vec![2, 4],
                vec![3, 4],
                vec![2, 3],
                vec![1, 2],
                vec![1, 3],
                vec![1, 4],
            ],
        ),
    ];

    for (i, (in1, in2, out1)) in cases.into_iter().enumerate() {
        let res = p3::combine(in1, in2);
        assert_eq!(res.len(), out1.len(), "case {} failed", i);
        let o = out1.into_iter().collect::<HashSet<_>>();
        for e in res {
            assert!(o.contains(e.as_slice()), "case {} failed", i);
        }
    }
}

#[test]
fn test_subsets() {
    let cases = vec![
        (
            vec![1, 2, 3],
            vec![
                vec![],
                vec![1],
                vec![2],
                vec![1, 2],
                vec![3],
                vec![1, 3],
                vec![2, 3],
                vec![1, 2, 3],
            ],
        ),
        (vec![0], vec![vec![], vec![0]]),
    ];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        let res = p3::subsets(in1);
        assert_eq!(res.len(), out1.len(), "case {} failed", i);
        let o = out1.into_iter().collect::<HashSet<_>>();
        for e in res {
            assert!(o.contains(e.as_slice()), "case {} failed", i);
        }
    }
}

#[test]
fn test_exist() {
    let cases = vec![
        (
            vec![
                vec!['A', 'B', 'C', 'E'],
                vec!['S', 'F', 'C', 'S'],
                vec!['A', 'D', 'E', 'E'],
            ],
            "ABCCED",
            true,
        ),
        (
            vec![
                vec!['A', 'B', 'C', 'E'],
                vec!['S', 'F', 'C', 'S'],
                vec!['A', 'D', 'E', 'E'],
            ],
            "SEE",
            true,
        ),
        (
            vec![
                vec!['A', 'B', 'C', 'E'],
                vec!['S', 'F', 'C', 'S'],
                vec!['A', 'D', 'E', 'E'],
            ],
            "ABCB",
            false,
        ),
    ];

    for (i, (in1, in2, out1)) in cases.into_iter().enumerate() {
        assert_eq!(p3::exist(in1, in2.to_string()), out1, "case {} failed", i);
    }
}

#[test]
fn test_remove_duplicates() {
    let cases = vec![
        (vec![1, 1, 1, 2, 2, 3], vec![1, 1, 2, 2, 3]),
        (vec![0, 0, 1, 1, 1, 1, 2, 3, 3], vec![0, 0, 1, 1, 2, 3, 3]),
    ];

    for (i, (mut in1, out1)) in cases.into_iter().enumerate() {
        p3::remove_duplicates(&mut in1);
        assert_eq!(in1, out1, "case {} failed", i);
    }
}

#[test]
fn test_search() {
    let cases = vec![
        (vec![2, 5, 6, 0, 0, 1, 2], 0, true),
        (vec![2, 5, 6, 0, 0, 1, 2], 3, false),
    ];

    for (i, (in1, in2, out1)) in cases.into_iter().enumerate() {
        assert_eq!(p3::search(in1, in2), out1, "case {} failed", i);
    }
}

#[test]
fn test_partition() {
    let cases = vec![
        (vec![1, 4, 3, 2, 5, 2], 3, vec![1, 2, 2, 4, 3, 5]),
        (vec![2, 1], 2, vec![1, 2]),
    ];

    for (i, (in1, in2, out1)) in cases.into_iter().enumerate() {
        let head = ListNode::from_slice(in1.as_slice());
        let out1 = ListNode::from_slice(out1.as_slice());
        assert_eq!(p3::partition(head, in2), out1, "case {} failed", i);
    }
}

#[test]
fn test_gray_code() {
    let cases = vec![(2, vec![0, 1, 3, 2]), (1, vec![0, 1])];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        assert_eq!(p3::gray_code(in1), out1, "case {} failed", i);
    }
}

#[test]
fn test_subsets_with_dup() {
    let cases = vec![
        (
            vec![1, 2, 2],
            vec![
                vec![],
                vec![1],
                vec![1, 2],
                vec![1, 2, 2],
                vec![2],
                vec![2, 2],
            ],
        ),
        (vec![0], vec![vec![], vec![0]]),
    ];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        let h = p3::subsets_with_dup(in1);
        assert_eq!(h.len(), out1.len(), "case {} failed", i);
        let h = h.into_iter().collect::<HashSet<_>>();
        for e in out1 {
            assert!(h.contains(e.as_slice()), "case {} failed", i);
        }
    }
}

#[test]
fn test_num_decodings() {
    let cases = vec![("12", 2), ("226", 3), ("0", 0)];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        assert_eq!(
            p3::num_decodings(in1.to_string()),
            out1,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_reverse_between() {
    let cases = vec![
        (vec![1, 2, 3, 4, 5], 2, 4, vec![1, 4, 3, 2, 5]),
        (vec![5], 1, 1, vec![5]),
    ];

    for (i, (in1, in2, in3, out1)) in cases.into_iter().enumerate() {
        assert_eq!(
            p3::reverse_between(ListNode::from_slice(in1.as_slice()), in2, in3),
            ListNode::from_slice(out1.as_slice()),
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_restore_ip_addresses() {
    let cases = vec![
        (
            "25525511135",
            vec![format!("255.255.11.135"), format!("255.255.111.35")],
        ),
        ("0000", vec![format!("0.0.0.0")]),
    ];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        assert_eq!(
            p3::restore_ip_addresses(in1.to_string()),
            out1,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_generate_trees() {
    let cases = vec![
        (
            3,
            vec![
                vec![1, i32::MAX, 2, i32::MAX, 3],
                vec![1, i32::MAX, 3, 2],
                vec![2, 1, 3],
                vec![3, 1, i32::MAX, i32::MAX, 2],
                vec![3, 2, i32::MAX, 1],
            ],
        ),
        (1, vec![vec![1]]),
    ];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        let trees = out1.into_iter().fold(Vec::new(), |mut trees, v| {
            trees.push(TreeNode::from_slice(v.as_slice()));
            trees
        });
        let o = p3::generate_trees(in1);
        assert_eq!(trees.len(), o.len(), "case {} failed", i);

        for e in o.iter() {
            assert!(trees.contains(e), "case {} failed", i);
        }
    }
}

#[test]
fn test_num_trees() {
    let cases = vec![(3, 5), (1, 1)];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        assert_eq!(p3::num_trees(in1), out1, "case {} failed", i);
    }
}

#[test]
fn test_is_interleave() {
    let cases = vec![
        ("aabcc", "dbbca", "aadbbcbcac", true),
        ("aabcc", "dbbca", "aadbbbaccc", false),
        ("", "", "", true),
    ];

    for (i, (in1, in2, in3, out1)) in cases.into_iter().enumerate() {
        assert_eq!(
            p3::is_interleave(in1.to_string(), in2.to_string(), in3.to_string()),
            out1,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_is_valid_bst() {
    let cases = vec![
        (vec![2, 1, 3], true),
        (vec![5, 1, 4, i32::MAX, i32::MAX, 3, 6], false),
    ];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        assert_eq!(
            p3::is_valid_bst(TreeNode::from_slice(in1.as_slice())),
            out1,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_recover_tree() {
    let cases = vec![
        (
            vec![1, 3, i32::MAX, i32::MAX, 2],
            vec![3, 1, i32::MAX, i32::MAX, 2],
        ),
        (
            vec![3, 1, 4, i32::MAX, i32::MAX, 2],
            vec![2, 1, 4, i32::MAX, i32::MAX, 3],
        ),
    ];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        let mut root = TreeNode::from_slice(in1.as_slice());
        p3::recover_tree(&mut root);
        let out1 = TreeNode::from_slice(out1.as_slice());
        assert_eq!(root, out1, "case {} failed", i);
    }
}
