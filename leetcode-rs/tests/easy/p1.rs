use leetcode_rs::easy::p1;
use leetcode_rs::easy::p1::{ListNode, TreeNode};

#[test]
fn test_two_sum() {
    let cases = vec![
        (vec![2, 7, 11, 15], 9, vec![0, 1]),
        (vec![3, 2, 4], 6, vec![1, 2]),
        (vec![3, 3], 6, vec![0, 1]),
    ];

    for (i, (input, target, output)) in cases.into_iter().enumerate() {
        assert_eq!(p1::two_sum(input, target), output, "case {} failed", i);
    }
}

#[test]
fn test_is_palindrome() {
    let cases = vec![(121, true), (-121, false), (10, false)];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(p1::is_palindrome(input), output, "case {} failed", i);
    }
}

#[test]
fn test_romain_to_int() {
    let cases = vec![
        ("II", 2),
        ("XII", 12),
        ("XXVII", 27),
        ("IV", 4),
        ("IX", 9),
        ("XL", 40),
        ("XC", 90),
        ("CD", 400),
        ("CM", 900),
        ("III", 3),
        ("LVIII", 58),
        ("MCMXCIV", 1994),
    ];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p1::roman_to_int(input.to_string()),
            output,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_longest_common_prefix() {
    let cases = vec![
        (vec!["flower", "flow", "flight"], "fl"),
        (vec![], ""),
        (vec!["dog", "rececar", "car"], ""),
        (vec!["haha"], "haha"),
    ];

    for (i, (input, output)) in cases
        .into_iter()
        .map(|ele| {
            (
                ele.0.into_iter().map(|e| e.to_string()).collect::<Vec<_>>(),
                ele.1,
            )
        })
        .enumerate()
    {
        assert_eq!(
            p1::longest_common_prefix(input),
            output,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_is_valid() {
    let cases = vec![("()", true), ("()[]{}", true), ("(]", false), ("", true)];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(p1::is_valid(input.to_string()), output, "case {} failed", i);
    }
}

#[test]
fn test_merge_two_lists() {
    let cases = vec![
        (vec![1, 2, 4], vec![1, 3, 4], vec![1, 1, 2, 3, 4, 4]),
        (vec![], vec![], vec![]),
        (vec![], vec![0], vec![0]),
    ];

    for (i, (list1, list2, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p1::merge_two_lists(
                ListNode::from_slice(list1.as_slice()),
                ListNode::from_slice(list2.as_slice())
            ),
            ListNode::from_slice(output.as_slice()),
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_remove_duplicates() {
    let cases = vec![
        (vec![1, 1, 2], vec![1, 2]),
        (vec![0, 0, 1, 1, 2, 2, 3, 3, 4], vec![0, 1, 2, 3, 4]),
        (vec![], vec![]),
        (vec![1], vec![1]),
    ];

    for (i, (mut input, output)) in cases.into_iter().enumerate() {
        p1::remove_duplicates(&mut input);
        assert_eq!(input, output, "case {} failed", i);
    }
}

#[test]
fn test_remove_element() {
    let cases = vec![
        ((vec![], 1), vec![]),
        ((vec![1], 2), vec![1]),
        ((vec![1], 1), vec![]),
        ((vec![1, 2], 1), vec![2]),
        ((vec![1, 1, 1], 1), vec![]),
        ((vec![3, 2, 2, 3], 3), vec![2, 2]),
        ((vec![0, 1, 2, 2, 3, 0, 4, 2], 2), vec![0, 1, 4, 0, 3]),
    ];

    for (i, ((mut input, target), output)) in cases.into_iter().enumerate() {
        p1::remove_element(&mut input, target);
        assert_eq!(input, output, "case {} failed", i);
    }
}

#[test]
fn test_str_str() {
    let cases = vec![
        (("hello", "ll"), 2),
        (("aaaaa", "bba"), -1),
        (("", ""), 0),
        (("a", "abb"), -1),
    ];

    for (i, ((src, dst), output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p1::str_str(src.to_string(), dst.to_string()),
            output,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_search_insert() {
    let cases = vec![
        ((vec![1, 3, 5, 6], 5), 2),
        ((vec![1, 3, 5, 6], 2), 1),
        ((vec![1, 3, 5, 6], 7), 4),
        ((vec![], 2), 0),
    ];
    for (i, ((input, target), output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p1::search_insert(input, target),
            output,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_max_sub_array() {
    let cases = vec![
        (vec![-2, 1, -3, 4, -1, 2, 1, -5, 4], 6),
        (vec![1], 1),
        (vec![5, 4, -1, 7, 8], 23),
    ];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(p1::max_sub_array(input), output, "case {} failed", i);
    }
}

#[test]
fn test_length_of_last_word() {
    let cases = vec![
        ("Hello World", 5),
        ("   fly me   to   the moon  ", 4),
        ("luffy is still joyboy", 6),
    ];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p1::length_of_last_word(input.to_string()),
            output,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_plus_one() {
    let cases = vec![
        (vec![1, 2, 3], vec![1, 2, 4]),
        (vec![4, 3, 2, 1], vec![4, 3, 2, 2]),
        (vec![0], vec![1]),
        (vec![9], vec![1, 0]),
    ];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(p1::plus_one(input), output, "case {} failed", i);
    }
}

#[test]
fn test_add_binary() {
    let cases = vec![
        (("11", "1"), "100"),
        (("1010", "1011"), "10101"),
        (("0", "0"), "0"),
        (("0", "1"), "1"),
    ];

    for (i, ((a, b), output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p1::add_binary(a.to_string(), b.to_string()),
            output,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_my_sqrt() {
    for i in 0..1000 {
        let output = (i as f64).sqrt().floor() as i32;
        assert_eq!(p1::my_sqrt(i), output, "case {} failed", i);
    }

    for (idx, i) in ((i32::MAX >> 1)..((i32::MAX >> 1) + 1000)).enumerate() {
        let output = (i as f64).sqrt().floor() as i32;
        assert_eq!(p1::my_sqrt(i), output, "case {} failed", idx);
    }

    for (idx, i) in ((i32::MAX - 10001)..i32::MAX).enumerate() {
        let output = (i as f64).sqrt().floor() as i32;
        assert_eq!(p1::my_sqrt(i), output, "case {} failed", idx);
    }
}

#[test]
fn test_climb_stairs() {
    let cases = [(1, 1), (2, 2), (3, 3)];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(p1::climb_stairs(input), output, "case {} failed", i);
    }
}

#[test]
fn test_delete_duplicates() {
    let cases = vec![
        (vec![1, 1, 2], vec![1, 2]),
        (vec![1, 1, 2, 3, 3], vec![1, 2, 3]),
        (vec![], vec![]),
    ];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        let (input, output) = (
            ListNode::from_slice(input.as_slice()),
            ListNode::from_slice(output.as_slice()),
        );
        assert_eq!(p1::delete_duplicates(input), output, "case {} failed", i);
    }
}

#[test]
fn test_merge() {
    let cases = vec![
        (vec![], vec![], vec![]),
        (vec![], vec![2], vec![2]),
        (vec![1, 2, 3], vec![2, 5, 6], vec![1, 2, 2, 3, 5, 6]),
        (vec![1], vec![], vec![1]),
    ];

    for (i, (mut i1, mut i2, o)) in cases.into_iter().enumerate() {
        let (m, n) = (i1.len() as i32, i2.len() as i32);
        p1::merge(&mut i1, m, &mut i2, n);
        assert_eq!(i1, o, "case {} failed", i);
    }
}

#[test]
fn test_inorder_traversal() {
    let cases = vec![
        (vec![1, i32::MAX, 2, 3], vec![1, 3, 2]),
        (vec![], vec![]),
        (vec![1], vec![1]),
    ];

    for (i, (input, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p1::inorder_traversal(TreeNode::from_slice(input.as_slice())),
            output,
            "case {} failed",
            i
        );
    }
}

#[test]
fn test_is_same_tree() {
    let cases = vec![
        (vec![1, 2, 3], vec![1, 2, 3], true),
        (vec![1, 2], vec![1, i32::MAX, 2], false),
        (vec![], vec![], true),
        (vec![1], vec![], false),
        (vec![1, 2, 1], vec![1, 1, 2], false),
    ];

    for (i, (in1, in2, output)) in cases.into_iter().enumerate() {
        assert_eq!(
            p1::is_same_tree(
                TreeNode::from_slice(in1.as_slice()),
                TreeNode::from_slice(in2.as_slice())
            ),
            output,
            "case {} failed",
            i
        );
    }
}
