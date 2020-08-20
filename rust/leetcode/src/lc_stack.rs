use std::str::FromStr;
use std::string::ToString;

/// Longest Valid Parentheses
/// 
/// Given a string containing just the characters ’(’ and ’)’, find the length of the longest valid (wellformed) parentheses substring.
/// For ”(()”, the longest valid parentheses substring is ”()”, which has length = 2.
/// Another example is ”)()())”, where the longest valid parentheses substring is ”()()”, which has
/// length = 4.
pub fn longest_valid_parentheses(s: String) -> i32 {
    
    let mut v = Vec::with_capacity(s.len() >> 1);
    let mut pos = -1;
    
    s.chars().enumerate().fold(0, move |longest, (idx, cur)| {
        let idx = idx as i32;
        if cur == '(' {
            v.push(idx);
            longest
        } else {
            match v.pop() { 
                Some(_) => {
                    match v.last() {
                        Some(&x) => {
                            std::cmp::max(longest, idx - x)
                        },
                        None => {
                            std::cmp::max(longest, idx - pos)
                        },
                    }
                },
                None => {
                    pos = idx;
                    longest
                },
            }
        }
    })
}

/// Largest Rectangle in Histogram
/// 
/// Given n non-negative integers representing the histogram’s bar height where the width of each bar is
/// 1, find the area of largest rectangle in the histogram.
pub fn largest_rectangle_area(height: Vec<i32>) -> i32 {
    let mut stk = Vec::with_capacity(height.len());
    let (mut i, mut result) = (0, 0);
    
    while i < height.len() {
        if stk.is_empty() || height[i] > height[*stk.last().unwrap()] {
            stk.push(i);
            i += 1;
        } else {
            let tmp = stk.pop().unwrap();
            result = std::cmp::max(result, (height[tmp] as usize) * match stk.last() {
                Some(&x) => i - x -1,
                None => i,
            });
        }
    }
    
    (0..stk.len()).for_each(|_| {
        let tmp = stk.pop().unwrap();
        result = std::cmp::max(result, (height[tmp] as usize) * match stk.last() {
            Some(&x) => i - x -1,
            None => i,
        });
    });

    result as i32
}

/// Evaluate Reverse Polish Notation
/// 
/// Evaluate the value of an arithmetic expression in Reverse Polish Notation.
/// Valid operators are +, -, *, /. Each operand may be an integer or another expression.
/// Some examples:
/// ["2", "1", "+", "3", "*"] -> ((2 + 1) * 3) -> 9
/// ["4", "13", "5", "/", "+"] -> (4 + (13 / 5)) -> 6
pub fn eval_rpn(tokens: Vec<String>) -> i32 {
    
    let mut stk = Vec::with_capacity(tokens.len() >> 2);
    tokens.iter().for_each(|x| {
        if (x.len() != 1) || ("+-*/".find(x.as_str()).is_none()) {
            stk.push(x.to_string())
        } else {
            let b = i32::from_str(stk.pop().unwrap().as_str()).unwrap();
            let a = i32::from_str(stk.pop().unwrap().as_str()).unwrap();
            let res = if x == "+" {
                a+b
            } else if x=="-" {
                a-b
            } else if x== "*" {
                a *b
            } else {
                a/b
            };
            stk.push(res.to_string());
        }
    });
    
    i32::from_str(stk.last().unwrap().as_str()).unwrap()
}

#[cfg(test)]
mod test {
    #[test] 
    fn longest_valid_parentheses() {
        let cases = [
            ("(()", 2),
            (")()())", 4),
            ("()()())()",6),
        ];

        cases.iter().for_each(|x| {
            let tmp = super::longest_valid_parentheses(x.0.to_string());
            assert_eq!(x.1, tmp, "case: {:?}", x.0);
        })
    }
    
    #[test]
    fn largest_rectangle_area() {

        let cases = [
            (vec![0,0],0),
            (vec![2,1,5,6,2,3], 10),
            (vec![7,4,2,3,4,5,3,2,3,3],20),
        ];

        cases.iter().for_each(|x| {
            let tmp = super::largest_rectangle_area(x.0.clone());
            assert_eq!(x.1,tmp, "case: {:?}", x.0);
        })
    }
    
    #[test]
    fn eval_rpn() {

        let cases = [
            (vec![String::from("2"),String::from("1"),String::from("+"),String::from("3"),String::from("*")], 9),
            (vec![String::from("4"),String::from("13"),String::from("5"),String::from("/"),String::from("+")], 6),
        ];

        cases.iter().for_each(|x| {
            let tmp = super::eval_rpn(x.0.to_vec());
            assert_eq!(x.1, tmp, "case: {:?}", x.0);
        })
    }
}