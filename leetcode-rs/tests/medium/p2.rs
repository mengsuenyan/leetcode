use leetcode_rs::easy::p1::ListNode;
use leetcode_rs::medium::p2;
use std::collections::HashSet;
use std::ops::Sub;

#[test]
fn test_count_and_say() {
    let cases = vec![
        "",
        "1",
        "11",
        "21",
        "1211",
        "111221",
        "312211",
        "13112221",
        "1113213211",
        "31131211131221",
        "13211311123113112211",
        "11131221133112132113212221",
        "3113112221232112111312211312113211",
        "1321132132111213122112311311222113111221131221",
    ];

    for (i, output) in cases.into_iter().enumerate().skip(1) {
        assert_eq!(p2::count_and_say(i as i32), output, "case {} failed", i);
    }
}

#[test]
fn test_combination_sum() {
    let cases = vec![
        (vec![2, 3, 6, 7], 7, vec![vec![2, 2, 3], vec![7]]),
        (vec![2], 1, vec![]),
    ];

    for (i, (input, tgt, output)) in cases.into_iter().enumerate() {
        let t = p2::combination_sum(input, tgt);
        assert_eq!(t.len(), output.len(), "case {} failed", i);
        for o in output {
            assert!(t.contains(&o), "case {} failed", i);
        }
    }
}

#[test]
fn test_combination_sum2() {
    let cases = vec![
        (
            vec![10, 1, 2, 7, 6, 1, 5],
            8,
            vec![vec![1, 1, 6], vec![1, 2, 5], vec![1, 7], vec![2, 6]],
        ),
        (vec![2], 1, vec![]),
        (vec![2, 5, 2, 1, 2], 5, vec![vec![1, 2, 2], vec![5]]),
    ];

    for (i, (input, tgt, output)) in cases.into_iter().enumerate() {
        let t = p2::combination_sum2(input, tgt);
        assert_eq!(t.len(), output.len(), "case {} failed", i);
        for o in output {
            assert!(t.contains(&o), " case {} failed", i);
        }
    }
}

#[test]
fn test_multiply() {
    let cases = vec![
        ("123", "456", "56088"),
        ("", "", ""),
        ("1", "2", "2"),
        ("2", "3", "6"),
    ];

    for (i, (in1, in2, out)) in cases.into_iter().enumerate() {
        assert_eq!(
            p2::multiply(in1.to_string(), in2.to_string()),
            out,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_jump() {
    let cases = vec![
        (vec![2, 3, 1, 1, 4], 2),
        (vec![1], 0),
        (vec![2, 3, 0, 1, 4], 2),
    ];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        assert_eq!(p2::jump(in1), out1, "case {} failed", i);
    }
}

#[test]
fn test_permute() {
    let cases = vec![
        (
            vec![1, 2, 3],
            vec![
                vec![1, 2, 3],
                vec![1, 3, 2],
                vec![2, 1, 3],
                vec![2, 3, 1],
                vec![3, 1, 2],
                vec![3, 2, 1],
            ],
        ),
        (vec![0, 1], vec![vec![0, 1], vec![1, 0]]),
        (vec![1], vec![vec![1]]),
    ];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        let t = p2::permute(in1);
        assert_eq!(t.len(), out1.len(), "case {} failed", i);
        for o in out1.iter() {
            assert!(t.contains(&o), "case {} failed", i);
        }
    }
}

#[test]
fn test_permute_unique() {
    let cases = vec![
        (
            vec![1, 1, 2],
            vec![vec![1, 1, 2], vec![1, 2, 1], vec![2, 1, 1]],
        ),
        (
            vec![1, 2, 3],
            vec![
                vec![1, 2, 3],
                vec![1, 3, 2],
                vec![2, 1, 3],
                vec![2, 3, 1],
                vec![3, 1, 2],
                vec![3, 2, 1],
            ],
        ),
    ];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        let t = p2::permute_unique(in1);
        assert_eq!(t.len(), out1.len(), "case {} failed", i);
        for o in out1.iter() {
            assert!(t.contains(&o), "case {} failed", i);
        }
    }
}

#[test]
fn test_rotate() {
    let cases = vec![
        (
            vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]],
            vec![vec![7, 4, 1], vec![8, 5, 2], vec![9, 6, 3]],
        ),
        (
            vec![
                vec![5, 1, 9, 11],
                vec![2, 4, 8, 10],
                vec![13, 3, 6, 7],
                vec![15, 14, 12, 16],
            ],
            vec![
                vec![15, 13, 2, 5],
                vec![14, 3, 4, 1],
                vec![12, 6, 8, 9],
                vec![16, 7, 10, 11],
            ],
        ),
    ];

    for (i, (mut in1, out1)) in cases.into_iter().enumerate() {
        p2::rotate(&mut in1);
        assert_eq!(in1, out1, "case {} failed", i);
    }
}

#[test]
fn test_group_anagrams() {
    let cases = vec![
        (
            vec!["eat", "tea", "tan", "ate", "nat", "bat"],
            vec![vec!["bat"], vec!["nat", "tan"], vec!["ate", "eat", "tea"]],
        ),
        (vec![""], vec![vec![""]]),
        (vec!["a"], vec![vec!["a"]]),
    ];
    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        let in1 = in1.into_iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let out1 = out1
            .into_iter()
            .map(|x| x.into_iter().map(|y| y.to_string()).collect::<Vec<_>>())
            .collect::<HashSet<_>>();
        let res = p2::group_anagrams(in1)
            .into_iter()
            .collect::<HashSet<Vec<String>>>();
        assert_eq!(res.len(), out1.len(), "case {} failed", i);
    }
}

#[test]
fn test_my_pow() {
    let cases = vec![
        (2f64, 10, 1024f64),
        (2.1f64, 3, 9.261f64),
        (2f64, -2, 0.25f64),
    ];

    for (i, (in1, tgt, out1)) in cases.into_iter().enumerate() {
        assert!(
            p2::my_pow(in1, tgt).sub(out1).abs() < 0.00000000000001,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_spiral_order() {
    let cases = vec![
        (
            vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]],
            vec![1, 2, 3, 6, 9, 8, 7, 4, 5],
        ),
        (
            vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 10, 11, 12]],
            vec![1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7],
        ),
    ];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        assert_eq!(p2::spiral_order(in1), out1, "case {} failed", i);
    }
}

#[test]
fn test_can_jump() {
    let cases = vec![
        (vec![2, 3, 1, 1, 4], true),
        (vec![], true),
        (vec![1], true),
        (vec![3, 2, 1, 0, 4], true),
    ];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        assert_eq!(p2::can_jump(in1), out1, "case {} failed", i);
    }
}

#[test]
fn test_merge() {
    let cases = vec![
        (
            vec![vec![1, 3], vec![2, 6], vec![8, 10], vec![15, 18]],
            vec![vec![1, 6], vec![8, 10], vec![15, 18]],
        ),
        (vec![vec![1, 4], vec![4, 5]], vec![vec![1, 5]]),
    ];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        assert_eq!(p2::merge(in1), out1, "case {} failed", i);
    }
}

#[test]
fn test_insert() {
    let cases = vec![
        (
            vec![vec![1, 3], vec![6, 9]],
            vec![2, 5],
            vec![vec![1, 5], vec![6, 9]],
        ),
        (
            vec![
                vec![1, 2],
                vec![3, 5],
                vec![6, 7],
                vec![8, 10],
                vec![12, 16],
            ],
            vec![4, 8],
            vec![vec![1, 2], vec![3, 10], vec![12, 16]],
        ),
    ];

    for (i, (in1, range, out1)) in cases.into_iter().enumerate() {
        assert_eq!(p2::insert(in1, range), out1, "case {} failed", i);
    }
}

#[test]
fn test_generate_matrix() {
    let cases = vec![
        (3, vec![vec![1, 2, 3], vec![8, 9, 4], vec![7, 6, 5]]),
        (1, vec![vec![1]]),
    ];
    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        assert_eq!(p2::generate_matrix(in1), out1, "case {} failed", i);
    }
}

#[test]
fn test_rotate_right() {
    let cases = vec![
        (vec![1, 2, 3, 4, 5], 2, vec![4, 5, 1, 2, 3]),
        (vec![0, 1, 2], 4, vec![2, 0, 1]),
    ];

    for (i, (in1, tgt, out1)) in cases.into_iter().enumerate() {
        assert_eq!(
            p2::rotate_right(ListNode::from_slice(in1.as_slice()), tgt),
            ListNode::from_slice(out1.as_slice()),
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_unique_paths() {
    let cases = vec![(3, 7, 28), (3, 2, 3), (7, 3, 28), (3, 3, 6)];

    for (i, (m, n, out1)) in cases.into_iter().enumerate() {
        assert_eq!(p2::unique_paths(m, n), out1, "case {} failed", i);
    }
}

#[test]
fn test_unique_paths_with_obstacles() {
    let cases = vec![
        (vec![vec![0, 0, 0], vec![0, 1, 0], vec![0, 0, 0]], 2),
        (vec![vec![0, 1], vec![0, 0]], 1),
    ];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        assert_eq!(
            p2::unique_paths_with_obstacles(in1),
            out1,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_min_path_sum() {
    let cases = vec![
        (vec![vec![1, 3, 1], vec![1, 5, 1], vec![4, 2, 1]], 7),
        (vec![vec![1, 2, 3], vec![4, 5, 6]], 12),
    ];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        assert_eq!(p2::min_path_sum(in1), out1, "case {} failed", i);
    }
}

#[test]
fn test_simplify_path() {
    let cases = vec![
        ("/home/", "/home"),
        ("/../", "/"),
        ("/home//foo/", "/home/foo"),
        ("/a/./b/../../c/", "/c"),
    ];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        assert_eq!(
            p2::simplify_path(in1.to_string()),
            out1,
            "case {} failed",
            i
        );
    }
}
