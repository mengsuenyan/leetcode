use leetcode_rs::easy::p1::ListNode;
use leetcode_rs::hard::p1;

#[test]
fn test_find_median_sorted_arrays() {
    let cases = vec![
        (vec![1, 3], vec![2], 2.0f64),
        (vec![1, 2], vec![3, 4], 2.5f64),
    ];

    for (i, (in1, in2, out1)) in cases.into_iter().enumerate() {
        let t = p1::find_median_sorted_arrays(in1, in2);

        assert!((t - out1).abs() < 0.0000001f64, "case {} failed", i);
    }
}

#[test]
fn test_is_match() {
    let cases = vec![("ab", ".*", true), ("aa", "a*", true), ("aa", "a", false)];

    for (i, (in1, in2, out1)) in cases.into_iter().enumerate() {
        assert_eq!(
            p1::is_match(in1.to_string(), in2.to_string()),
            out1,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_merge_k_lists() {
    let cases = vec![
        (
            vec![vec![1, 4, 5], vec![1, 3, 4], vec![2, 6]],
            vec![1, 1, 2, 3, 4, 4, 5, 6],
        ),
        (vec![vec![]], vec![]),
        (vec![], vec![]),
    ];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        let lists = in1
            .into_iter()
            .map(|x| ListNode::from_slice(x.as_slice()))
            .collect::<Vec<_>>();
        assert_eq!(
            p1::merge_k_lists(lists),
            ListNode::from_slice(out1.as_slice()),
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_find_substring() {
    let cases = vec![
        ("barfoothefoobarman", vec!["foo", "bar"], vec![0, 9]),
        (
            "wordgoodgoodgoodbestword",
            vec!["word", "good", "best", "word"],
            vec![],
        ),
        (
            "barfoofoobarthefoobarman",
            vec!["bar", "foo", "the"],
            vec![6, 9, 12],
        ),
    ];

    for (i, (in1, in2, out1)) in cases.into_iter().enumerate() {
        assert_eq!(
            p1::find_substring(
                in1.to_string(),
                in2.into_iter().map(|x| x.to_string()).collect()
            ),
            out1,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_longest_valid_parentheses() {
    let cases = vec![("(()", 2), (")()())", 4), ("", 0)];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        assert_eq!(
            p1::longest_valid_parentheses(in1.to_string()),
            out1,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_solve_sudoku() {
    let cases = vec![(
        vec![
            vec!['5', '3', '.', '.', '7', '.', '.', '.', '.'],
            vec!['6', '.', '.', '1', '9', '5', '.', '.', '.'],
            vec!['.', '9', '8', '.', '.', '.', '.', '6', '.'],
            vec!['8', '.', '.', '.', '6', '.', '.', '.', '3'],
            vec!['4', '.', '.', '8', '.', '3', '.', '.', '1'],
            vec!['7', '.', '.', '.', '2', '.', '.', '.', '6'],
            vec!['.', '6', '.', '.', '.', '.', '2', '8', '.'],
            vec!['.', '.', '.', '4', '1', '9', '.', '.', '5'],
            vec!['.', '.', '.', '.', '8', '.', '.', '7', '9'],
        ],
        vec![
            vec!['5', '3', '4', '6', '7', '8', '9', '1', '2'],
            vec!['6', '7', '2', '1', '9', '5', '3', '4', '8'],
            vec!['1', '9', '8', '3', '4', '2', '5', '6', '7'],
            vec!['8', '5', '9', '7', '6', '1', '4', '2', '3'],
            vec!['4', '2', '6', '8', '5', '3', '7', '9', '1'],
            vec!['7', '1', '3', '9', '2', '4', '8', '5', '6'],
            vec!['9', '6', '1', '5', '3', '7', '2', '8', '4'],
            vec!['2', '8', '7', '4', '1', '9', '6', '3', '5'],
            vec!['3', '4', '5', '2', '8', '6', '1', '7', '9'],
        ],
    )];

    for (i, (mut in1, out1)) in cases.into_iter().enumerate() {
        p1::solve_sudoku(&mut in1);
        assert_eq!(in1, out1, "case {} failed", i);
    }
}

#[test]
fn test_first_missing_positive() {
    let cases = vec![
        (vec![1, 2, 0], 3),
        (vec![3, 4, -1, 1], 2),
        (vec![7, 8, 9, 11, 12], 1),
    ];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        assert_eq!(p1::first_missing_positive(in1), out1, "case {} failed", i);
    }
}

#[test]
fn test_trap() {
    let cases = vec![
        (vec![0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1], 6),
        (vec![], 0),
        (vec![1], 0),
        (vec![4, 2, 0, 3, 2, 5], 9),
    ];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        assert_eq!(p1::trap(in1), out1, "case {} failed", i);
    }
}

#[test]
fn test_is_match_ii() {
    let cases = vec![
        ("aa", "a", false),
        ("aa", "*", true),
        ("cb", "?a", false),
        ("adceb", "*a*b", true),
        ("", "", true),
    ];

    for (i, (s, p, out1)) in cases.into_iter().enumerate() {
        assert_eq!(
            p1::is_match_ii(s.to_string(), p.to_string()),
            out1,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_solve_n_queens() {
    let cases = vec![(
        4,
        vec![
            vec![
                format!(".Q.."),
                format!("...Q"),
                format!("Q..."),
                format!("..Q."),
            ],
            vec![
                format!("..Q."),
                format!("Q..."),
                format!("...Q"),
                format!(".Q.."),
            ],
        ],
    )];

    for (i, (n, out1)) in cases.into_iter().enumerate() {
        assert_eq!(p1::solve_n_queens(n), out1, "case {} failed", i);
    }
}

#[test]
fn test_total_n_queens() {
    let cases = vec![(4, 2), (1, 1)];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        assert_eq!(p1::total_n_queens(in1), out1, "case {} failed", i);
    }
}

#[test]
fn test_get_permutation() {
    let cases = vec![(3, 1, "123"), (3, 3, "213"), (4, 9, "2314"), (1, 1, "1")];

    for (i, (n, k, out1)) in cases.into_iter().enumerate() {
        assert_eq!(p1::get_permutation(n, k), out1, "case {} failed", i);
    }
}

#[test]
fn test_is_number() {
    let cases = vec![
        (
            vec![
                "2e10",
                "2",
                "0089",
                "-0.1",
                "+3.14",
                "4.",
                "-.9",
                "-90E3",
                "3e+7",
                "+6e-1",
                "53.5e93",
                "-123.456e789",
            ],
            true,
        ),
        (
            vec!["abc", "1a", "1e", "e3", "99e2.5", "--6", "-+3", "95a54e53"],
            false,
        ),
    ];

    for (in1, out1) in cases.into_iter() {
        for s in in1 {
            assert_eq!(p1::is_number(s.to_string()), out1, "case {} failed", s);
        }
    }
}

#[test]
fn test_full_justify() {
    let cases = vec![
        (
            vec![
                format!("This"),
                format!("is"),
                format!("an"),
                format!("example"),
                format!("of"),
                format!("text"),
                format!("justification."),
            ],
            16,
            vec![
                format!("This    is    an"),
                format!("example  of text"),
                format!("justification.  "),
            ],
        ),
        (
            vec![
                format!("What"),
                format!("must"),
                format!("be"),
                format!("acknowledgment"),
                format!("shall"),
                format!("be"),
            ],
            16,
            vec![
                format!("What   must   be"),
                format!("acknowledgment  "),
                format!("shall be        "),
            ],
        ),
        (
            vec![
                format!("Science"),
                format!("is"),
                format!("what"),
                format!("we"),
                format!("understand"),
                format!("well"),
                format!("enough"),
                format!("to"),
                format!("explain"),
                format!("to"),
                format!("a"),
                format!("computer."),
                format!("Art"),
                format!("is"),
                format!("everything"),
                format!("else"),
                format!("we"),
                format!("do"),
            ],
            20,
            vec![
                format!("Science  is  what we"),
                format!("understand      well"),
                format!("enough to explain to"),
                format!("a  computer.  Art is"),
                format!("everything  else  we"),
                format!("do                  "),
            ],
        ),
    ];

    for (i, (words, max_width, out1)) in cases.into_iter().enumerate() {
        assert_eq!(
            p1::full_justify(words, max_width),
            out1,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_min_distance() {
    let cases = vec![
        ("horse", "row", 4),
        ("horse", "ros", 3),
        ("intention", "execution", 5),
    ];

    for (i, (in1, in2, out1)) in cases.into_iter().enumerate() {
        assert_eq!(
            p1::min_distance(in1.to_string(), in2.to_string()),
            out1,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_min_window() {
    let cases = vec![
        ("ADOBECODEBANC", "ABC", "BANC"),
        ("a", "a", "a"),
        ("a", "aa", ""),
    ];

    for (i, (in1, in2, out1)) in cases.into_iter().enumerate() {
        assert_eq!(
            p1::min_window(in1.to_string(), in2.to_string()),
            out1,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_largest_rectangle_area() {
    let cases = vec![(vec![2, 1, 5, 6, 2, 3], 10), (vec![2, 4], 4)];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        assert_eq!(p1::largest_rectangle_area(in1), out1, "case {} failed", i);
    }
}

#[test]
fn test_maximal_rectangle() {
    let cases = vec![
        (
            vec![
                vec!['1', '0', '1', '0', '0'],
                vec!['1', '0', '1', '1', '1'],
                vec!['1', '1', '1', '1', '1'],
                vec!['1', '0', '0', '1', '0'],
            ],
            6,
        ),
        (vec![vec![]], 0),
        (vec![vec!['0']], 0),
        (vec![vec!['1']], 1),
        (vec![vec!['0', '0']], 0),
    ];

    for (i, (in1, out1)) in cases.into_iter().enumerate() {
        assert_eq!(p1::maximal_rectangle(in1), out1, "case {} failed", i);
    }
}

#[test]
fn test_is_scramble() {
    let cases = vec![("abcde", "caebd", false), ("a", "a", true)];

    for (i, (in1, in2, out1)) in cases.into_iter().enumerate() {
        assert_eq!(
            p1::is_scramble(in1.to_string(), in2.to_string()),
            out1,
            "case {} failed",
            i
        );
    }
}
