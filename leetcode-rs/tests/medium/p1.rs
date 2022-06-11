use leetcode_rs::easy::p1::ListNode;
use leetcode_rs::medium::p1;
use std::collections::HashSet;

#[test]
fn test_add_two_numbers() {
    let cases = vec![
        (vec![2, 4, 3], vec![5, 6, 4], vec![7, 0, 8]),
        (vec![0], vec![0], vec![0]),
        (
            vec![9, 9, 9, 9, 9, 9, 9],
            vec![9, 9, 9, 9],
            vec![8, 9, 9, 9, 0, 0, 0, 1],
        ),
    ];

    for (i, (l1, l2, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p1::add_two_numbers(
                ListNode::from_slice(l1.as_slice()),
                ListNode::from_slice(l2.as_slice())
            ),
            ListNode::from_slice(output.as_slice()),
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_length_of_longest_substring() {
    let cases = vec![("pwwkew", 3), ("abcabcbb", 3), ("bbbbb", 1), ("", 0)];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p1::length_of_longest_substring(input.to_string()),
            output,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_longest_palindrome() {
    let cases = vec![("", ""), ("babad", "aba"), ("cbbd", "bb")];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p1::longest_palindrome(input.to_string()),
            output,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_convert() {
    let cases = vec![
        ("PAYPALISHIRING", 3, "PAHNAPLSIIGYIR"),
        ("PAYPALISHIRING", 4, "PINALSIGYAHRPI"),
        ("", 1, ""),
    ];

    for (i, (input, row, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p1::convert(input.to_string(), row),
            output,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_reverse() {
    let cases = vec![(0, 0), (120, 21), (-123, -321), (123, 321)];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(p1::reverse(input), output, "case {} failed", i);
    }
}

#[test]
fn test_my_atoi() {
    let cases = vec![
        ("       -42", -42),
        ("", 0),
        ("42", 42),
        ("4193 with words", 4193),
    ];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(p1::my_atoi(input.to_string()), output, "case {} failed", i);
    }
}

#[test]
fn test_max_area() {
    let cases = vec![
        (vec![1, 8, 6, 2, 5, 4, 8, 3, 7], 49),
        (vec![1, 1], 1),
        (vec![], 0),
    ];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(p1::max_area(input), output, "case {} failed", i);
    }
}

#[test]
fn test_int_to_roman() {
    let cases = vec![(1994, "MCMXCIV"), (58, "LVIII"), (3, "III")];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(p1::int_to_roman(input), output, "case {} failed", i);
    }
}

#[test]
fn test_three_sum() {
    let cases = vec![
        (
            vec![-1, 0, 1, 2, -1, -4],
            vec![vec![-1, -1, 2], vec![-1, 0, 1]],
        ),
        (vec![], vec![]),
        (vec![0], vec![]),
    ];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        let h = p1::three_sum(input)
            .into_iter()
            .collect::<HashSet<Vec<i32>>>();
        assert_eq!(h.len(), output.len(), "case {} failed", i);
        for o in output {
            assert!(h.contains(o.as_slice()), "{:?} not in the output", o);
        }
    }
}

#[test]
fn test_three_sum_closest() {
    let cases = vec![(vec![-1, 2, 1, -4], 1, 2), (vec![0, 0, 0], 1, 0)];

    for (i, (input, target, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p1::three_sum_closest(input, target),
            output,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_letter_combinations() {
    let cases = vec![("23", 9), ("", 0), ("2", 3), ("278", 36)];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p1::letter_combinations(input.to_string()).len(),
            output,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_four_sum() {
    let cases = vec![
        (
            vec![1, 0, -1, 0, -2, 2],
            0,
            vec![vec![-2, -1, 1, 2], vec![-2, 0, 0, 2], vec![-1, 0, 0, 1]],
        ),
        (vec![2, 2, 2, 2], 8, vec![vec![2, 2, 2, 2]]),
        (vec![], 1, vec![]),
        (vec![1], 1, vec![]),
        (vec![1, 2], 3, vec![]),
        (vec![1, 2, 3], 6, vec![]),
        (vec![1, 2, 3, 4], 9, vec![]),
    ];

    for (i, (input, target, output)) in cases.into_iter().enumerate() {
        let h = p1::four_sum(input, target)
            .into_iter()
            .collect::<HashSet<Vec<_>>>();
        assert_eq!(h.len(), output.len(), "case {} failed", i);
        for o in output {
            assert!(h.contains(o.as_slice()), "case {:?} failed", o);
        }
    }
}

#[test]
fn test_remove_nth_from_end() {
    let cases = vec![
        (vec![1, 2, 3, 4, 5], 2, vec![1, 2, 3, 5]),
        (vec![1], 1, vec![]),
        (vec![1, 2], 1, vec![1]),
    ];
    for (i, (input, target, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p1::remove_nth_from_end(ListNode::from_slice(input.as_slice()), target),
            ListNode::from_slice(output.as_slice()),
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_generate_parenthesis() {
    let cases = vec![
        (1, vec!["()"]),
        (2, vec!["()()", "(())"]),
        (3, vec!["((()))", "(()())", "(())()", "()(())", "()()()"]),
    ];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        let h = p1::generate_parenthesis(input)
            .into_iter()
            .collect::<HashSet<String>>();
        assert_eq!(h.len(), output.len(), "case {} failed", i);
        for o in output {
            assert!(h.contains(o), "case {} failed", i);
        }
    }
}

#[test]
fn test_swap_pairs() {
    let cases = vec![
        (vec![1, 2, 3, 4], vec![2, 1, 4, 3]),
        (vec![], vec![]),
        (vec![1], vec![1]),
    ];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p1::swap_pairs(ListNode::from_slice(input.as_slice())),
            ListNode::from_slice(output.as_slice()),
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_divide() {
    let cases = vec![
        (7, -3, -2),
        (-10, 3, -3),
        (10, 3, 3),
        (-10, -3, 3),
        (i32::MAX, i32::MIN, 0),
        (i32::MIN, 1, i32::MIN),
        (0, 1, 0),
    ];

    for (i, (dividend, divisor, output)) in cases.into_iter().enumerate() {
        assert_eq!(p1::divide(dividend, divisor), output, "case {} failed", i);
    }
}

#[test]
fn test_next_permutation() {
    let cases = vec![
        (vec![1, 2, 3], vec![1, 3, 2]),
        (vec![3, 2, 1], vec![1, 2, 3]),
        (vec![1, 1, 5], vec![1, 5, 1]),
        (vec![], vec![]),
    ];

    for (i, (mut input, output)) in cases.into_iter().enumerate() {
        p1::next_permutation(input.as_mut());
        assert_eq!(input, output, "case {} failed", i);
    }
}

#[test]
fn test_search() {
    let cases = vec![
        (vec![4, 5, 6, 7, 0, 1, 2], 0, 4),
        (vec![], 0, -1),
        (vec![4, 5, 6, 7, 0, 1, 2], 3, -1),
        (vec![1], 0, -1),
    ];

    for (i, (input, target, output)) in cases.into_iter().enumerate() {
        assert_eq!(p1::search(input, target), output, "case {} failed", i);
    }
}

#[test]
fn test_search_range() {
    let cases = vec![
        (vec![5, 7, 7, 8, 8, 10], 6, vec![-1, -1]),
        (vec![5, 7, 7, 8, 8, 10], 8, vec![3, 4]),
        (vec![], 0, vec![-1, -1]),
    ];

    for (i, (input, target, output)) in cases.into_iter().enumerate() {
        assert_eq!(p1::search_range(input, target), output, "case {} failed", i);
    }
}

#[test]
fn test_is_valid_sudoku() {
    let cases = vec![
        (
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
            true,
        ),
        (
            vec![
                vec!['8', '3', '.', '.', '7', '.', '.', '.', '.'],
                vec!['6', '.', '.', '1', '9', '5', '.', '.', '.'],
                vec!['.', '9', '8', '.', '.', '.', '.', '6', '.'],
                vec!['8', '.', '.', '.', '6', '.', '.', '.', '3'],
                vec!['4', '.', '.', '8', '.', '3', '.', '.', '1'],
                vec!['7', '.', '.', '.', '2', '.', '.', '.', '6'],
                vec!['.', '6', '.', '.', '.', '.', '2', '8', '.'],
                vec!['.', '.', '.', '4', '1', '9', '.', '.', '5'],
                vec!['.', '.', '.', '.', '8', '.', '.', '7', '9'],
            ],
            false,
        ),
    ];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(p1::is_valid_sudoku(input), output, "case {} failed", i);
    }
}
