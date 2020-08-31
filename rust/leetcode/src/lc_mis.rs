use std::collections::HashMap;

pub fn my_pow(x: f64, n: i32) -> f64 {
    x.powi(n)
}

pub fn spiral_order(matrix: Vec<Vec<i32>>) -> Vec<i32> {
    if matrix.is_empty() {
        return vec![];
    }
    
    let mut ridx = matrix.last().unwrap().len();
    let col = ridx;
    let mut v = Vec::with_capacity(col * matrix.len());
    let mut lidx = 0;
    let num = matrix.len() / 2;
    let rem = matrix.len() % 2;
    let mut head= matrix.iter();
    let mut tail = matrix.iter().rev();
    
    let mut i = 0;
    while v.len() < col * matrix.len() && i < num {
        
        let ele = &(head.next().unwrap().as_slice())[lidx..ridx];
        let mut row = ele.to_vec();
        v.append(&mut row);
        
        let m = &(matrix.as_slice())[(i+1)..(matrix.len()-1-i)];
        for row in m.iter() {
            v.push(row[ridx - 1]);
        }

        let mut ele = (&(tail.next().unwrap().as_slice())[lidx..ridx]).to_vec();
        ele.reverse();
        v.append(&mut ele);
        
        if lidx < ridx - 1 {
            for row in m.iter().rev() {
                v.push(row[lidx]);
            }
            lidx += 1;
        }

        ridx -= 1;
        i +=1 ;
    }

    if rem > 0 {
        let ele = &(head.next().unwrap().as_slice())[lidx..ridx];
        let mut row = ele.to_vec();
        v.append(&mut row);
    }
    
    v
}

/// O(n)
/// ---上---
/// |  ... |
/// 左  ... 右
/// |---下---|
pub fn generate_matrix(n: i32) -> Vec<Vec<i32>> {
    if n < 1 {
        return vec![];
    }
    
    let mut v = vec![vec![0i32; n as usize]; n as usize];
    let mut lidx = 0;
    let mut ridx = (n as usize) - 1;
    let mut uidx = 0;
    let mut didx = (n as usize) - 1; 
    
    let n = n * n;
    let mut num = 0;
    loop {
        // 上
        let mut col = lidx;
        while col <= ridx {
            num += 1;
            v[uidx][col] = num;
            col += 1;
        }
        uidx += 1;
        if num >= n {break;}
        
        // 右
        let mut row = uidx;
        while row <= didx {
            num += 1;
            v[row][ridx] = num;
            row += 1;
        }
        ridx -= 1;
        if num >= n {break;}
        
        // 下
        let mut col = ridx;
        while col >= lidx {
            num += 1;
            v[didx][col] = num;
            if col == lidx {
                break;
            } else {
                col -= 1;
            }
        }
        didx -= 1;
        if num >= n {break;}
        
        // 左
        let mut row = didx;
        while row >= uidx {
            num += 1;
            v[row][lidx] = num;
            if row == uidx {
                break;
            } else {
                row -= 1;
            }
        }
        lidx += 1;
        if num >= n {break;}
    }
    
    v
}

pub fn my_sqrt(x: i32) -> i32 {
    let x = x as f64;
    x.sqrt().floor() as i32
}

/// Divide Two Integers
/// Given two integers dividend and divisor, divide two integers without using multiplication, division and mod operator.
/// 
/// Return the quotient after dividing dividend by divisor.
/// 
/// The integer division should truncate toward zero, which means losing its fractional part. For example, truncate(8.345) = 8 and truncate(-2.7335) = -2.
/// 
/// Example 1:
/// 
/// Input: dividend = 10, divisor = 3
/// Output: 3
/// Explanation: 10/3 = truncate(3.33333..) = 3.
/// 
/// Example 2:
/// 
/// Input: dividend = 7, divisor = -3
/// Output: -2
/// Explanation: 7/-3 = truncate(-2.33333..) = -2.
/// 
/// Note:
/// 
/// Both dividend and divisor will be 32-bit signed integers.
/// The divisor will never be 0.
/// Assume we are dealing with an environment which could only store integers within the 32-bit signed integer range: [−231,  231 − 1]. For the purpose of this problem, assume that your function returns 231 − 1 when the division result overflows.
/// 
/// 来源：力扣（LeetCode）
/// 链接：https://leetcode-cn.com/problems/divide-two-integers
/// 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
pub fn divide(dividend: i32, divisor: i32) -> i32 {
    use std::i32::*;
    let sw = |x: i32| {if x == MIN {1+(MAX as u32)} else {-x as u32}};
    
    let (dividend, mut divisor, is_pos) = if dividend < 0 && divisor < 0 {
        (sw(dividend), sw(divisor), true)
    } else if dividend < 0 && divisor > 0 {
        (sw(dividend), divisor as u32, false)
    } else if divisor < 0 && dividend > 0 {
        (dividend as u32, sw(divisor), false)
    } else {
        (dividend as u32, divisor as u32, true)
    };
    
    let mut res = if dividend < divisor {return 0;} 
        else if divisor == dividend {return if is_pos {1} else {-1};}
        else {1u32};

    let original = divisor;
    loop {
        
        match divisor << 1 {
            x if x < divisor => break,
            x => {
                if x <= dividend {
                    res <<= 1;
                    divisor = x;
                } else {
                    break;
                }
            }
        }
    }
    
    loop {
        match divisor.overflowing_add(original) { 
            (_, true) => break,
            (x, false) => {
                if x <= dividend {
                    res += 1;
                    divisor = x;
                } else {
                    break;
                }
            }
        }
    }
    
    if res > (MAX as u32) {if is_pos { MAX } else {MIN}} else {if is_pos {res as i32} else {-(res as i32)}}
}

/// Substring with Concatenation of All Words
/// You are given a string, s, and a list of words, words, that are all of the same length. 
/// Find all starting indices of substring(s) in s that is a concatenation of each word in 
/// words exactly once and without any intervening characters.
/// 
/// Example 1:
/// 
/// Input:
/// s = "barfoothefoobarman",
/// words = ["foo","bar"]
/// Output: [0,9]
/// Explanation: Substrings starting at index 0 and 9 are "barfoo" and "foobar" respectively.
/// The output order does not matter, returning [9,0] is fine too.
/// 
/// Example 2:
/// 
/// Input:
/// s = "wordgoodgoodgoodbestword",
/// words = ["word","good","best","word"]
/// Output: []
/// 
/// 来源：力扣（LeetCode）
/// 链接：https://leetcode-cn.com/problems/substring-with-concatenation-of-all-words
/// 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
pub fn find_substring(s: String, words: Vec<String>) -> Vec<i32> {
    let (len, word_len) = match words.first() {
        Some(x) => if x.len() * words.len() > s.len() {return vec![];} else {(x.len()*words.len(), x.len())},
        None => return vec![],
    };
    
    let mut h = HashMap::with_capacity(words.len());
    words.iter().for_each(|x| {
        *h.entry(x.clone()).or_insert(0) += 1;
    });
    
    let mut res = Vec::new();
    let s = s.as_str();
    for i in 0..(s.len() + 1 - len) {
        let mut hc = h.clone();
        let mut j = i;
        while j < (i + len) {
            let sub = &s[j..(j+word_len)];
            match hc.get_mut(sub) {
                Some(x) => {
                    if *x == 1 {
                        hc.remove(sub);
                    } else {
                        *x -= 1;
                    }
                },
                None => break,
            }
            j += word_len;
        }
        
        if hc.len() == 0 {
            res.push(i as i32);
        }
    }
    
    res
}

/// Multiply Strings
/// Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string.
/// 
/// Example 1:
/// 
/// Input: num1 = "2", num2 = "3"
/// Output: "6"
/// 
/// Example 2:
/// 
/// Input: num1 = "123", num2 = "456"
/// Output: "56088"
/// 
/// Note:
/// 
/// The length of both num1 and num2 is < 110.
/// Both num1 and num2 contain only digits 0-9.
/// Both num1 and num2 do not contain any leading zero, except the number 0 itself.
/// You must not use any built-in BigInteger library or convert the inputs to integer directly.
/// 
/// 来源：力扣（LeetCode）
/// 链接：https://leetcode-cn.com/problems/multiply-strings
/// 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
pub fn multiply(num1: String, num2: String) -> String {
    let n1: Vec<u32> = num1.chars().rev().map(|x| { (x as u32) - ('0' as u32)}).collect();
    let n2: Vec<u32> = num2.chars().rev().map(|x| { (x as u32) - ('0' as u32)}).collect();
    
    let mut m = vec![0; n1.len() + n2.len()];
    (0..n1.len()).for_each(|i| {
        (0..n2.len()).for_each(|j| {
            m[i+j] += n1[i]*n2[j];
            m[i+j+1] += m[i+j] / 10;
            m[i+j] %= 10;
        });
    });
    
    let mut s = String::with_capacity(m.len());
    m.iter().enumerate().rev().skip_while(|x| {
        x.0 != 0 && x.1 == &0
    }).for_each(|(_, &x)| {
        s.push((x as u8 + b'0') as char);
    });
    s
}

///  Group Anagrams
/// Given an array of strings strs, group the anagrams together. You can return the answer in any order.
/// 
/// An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.
/// 
/// 
/// 
/// Example 1:
/// 
/// Input: strs = ["eat","tea","tan","ate","nat","bat"]
/// Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
/// 
/// Example 2:
/// 
/// Input: strs = [""]
/// Output: [[""]]
/// 
/// Example 3:
/// 
/// Input: strs = ["a"]
/// Output: [["a"]]
/// 
/// 来源：力扣（LeetCode）
/// 链接：https://leetcode-cn.com/problems/group-anagrams
/// 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
pub fn group_anagrams(strs: Vec<String>) -> Vec<Vec<String>> {
    let mut h = HashMap::with_capacity(strs.len());
    strs.iter().for_each(|x| {
        let mut y: Vec<u32> = x.chars().map(|x| {x as u32}).collect();
        y.sort();
        h.entry(y).or_insert(Vec::new()).push(x.clone());
    });
    
    h.values().map(|x| {x.clone()}).collect()
}

/// Merge Intervals
/// Given a collection of intervals, merge all overlapping intervals.
/// 
/// Example 1:
/// 
/// Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
/// Output: [[1,6],[8,10],[15,18]]
/// Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].
/// 
/// Example 2:
/// 
/// Input: intervals = [[1,4],[4,5]]
/// Output: [[1,5]]
/// Explanation: Intervals [1,4] and [4,5] are considered overlapping.
/// 
/// NOTE: input types have been changed on April 15, 2019. Please reset to default code definition to get new method signature.
/// 
/// 
/// 
/// Constraints:
/// 
/// intervals[i][0] <= intervals[i][1]
/// 
/// 来源：力扣（LeetCode）
/// 链接：https://leetcode-cn.com/problems/merge-intervals
/// 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
pub fn merge(intervals: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let mut v = intervals;
    v.sort_by(|x,y| {x.first().unwrap().cmp(y.first().unwrap())});
    
    let mut e = std::i32::MIN;
    let mut res = Vec::new();
    
    v.iter().for_each(|x| {
        let (u, v) = (x[0], x[1]);
        if u > e {
            res.push(vec![u,v]);
            e = v;
        } else {
            let last = res.last_mut().unwrap().last_mut().unwrap();
            *last = std::cmp::max(v, *last);
            e = *last;
        }
    });
    
    res
}

/// Insert Interval
/// Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).
/// 
/// You may assume that the intervals were initially sorted according to their start times.
/// 
/// Example 1:
/// 
/// Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
/// Output: [[1,5],[6,9]]
/// 
/// Example 2:
/// 
/// Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
/// Output: [[1,2],[3,10],[12,16]]
/// Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].
/// 
/// NOTE: input types have been changed on April 15, 2019. Please reset to default code definition to get new method signature.
pub fn insert(intervals: Vec<Vec<i32>>, new_interval: Vec<i32>) -> Vec<Vec<i32>> {
    if new_interval.is_empty() { return intervals;}
    
    let (mut res, s, mut idx) = (Vec::new(), new_interval[0], intervals.len());
    let mut intervals = intervals;
    for x in intervals.iter().enumerate() {
        if s < (x.1)[0] {
            idx = x.0;
            break;
        }
    }
    
    intervals.insert(idx, new_interval);
    let mut e = std::i32::MIN;

    intervals.iter().for_each(|x| {
        let (u, v) = (x[0], x[1]);
        if u > e {
            res.push(vec![u,v]);
            e = v;
        } else {
            let last = res.last_mut().unwrap().last_mut().unwrap();
            *last = std::cmp::max(v, *last);
            e = *last;
        }
    });
    res
}

/// Text Justification
/// Given an array of words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.
/// 
/// You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces ' ' when necessary so that each line has exactly maxWidth characters.
/// 
/// Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.
/// 
/// For the last line of text, it should be left justified and no extra space is inserted between words.
/// 
/// Note:
/// 
/// A word is defined as a character sequence consisting of non-space characters only.
/// Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
/// The input array words contains at least one word.
/// 
/// Example 1:
/// 
/// Input:
/// words = ["This", "is", "an", "example", "of", "text", "justification."]
/// maxWidth = 16
/// Output:
/// [
/// "This    is    an",
/// "example  of text",
/// "justification.  "
/// ]
/// 
/// 来源：力扣（LeetCode）
/// 链接：https://leetcode-cn.com/problems/text-justification
/// 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
pub fn full_justify(words: Vec<String>, max_width: i32) -> Vec<String> {
    let add_space = |s: &mut String, i: usize, n: usize, l: usize, is_last| {
        if n < 1 || i > (n-1) {return;}
        let num = if is_last {1} else {l/n + if i < (l%n) {1} else {0}};
        s.push_str(" ".repeat(num).as_str());
    };
    
    let connect = |words: &Vec<String>, begin: usize, end: usize, len: usize, l: usize, is_last: bool| {
        let mut s = String::new();
        let n = end + 1 - begin;
        words.iter().skip(begin).take(n).enumerate().for_each(|x| {
            s.push_str(x.1.as_str());
            add_space(&mut s, x.0, n-1, l-len, is_last);
        });
        if s.len() < l {s.push_str(" ".repeat(l-s.len()).as_str());};
        s
    };
    
    let max_width = max_width as usize;
    let mut res = Vec::new();
    let (mut begin, mut len) = (0, 0);
    words.iter().enumerate().for_each(|x| {
        if len + (x.1).len() + (x.0 - begin) > max_width {
            res.push(connect(&words, begin, x.0-1, len, max_width, false));
            begin = x.0;
            len = 0;
        }
        len += (x.1).len();
    });
    res.push(connect(&words, begin, words.len()-1,len, max_width, true));
    res
}

/// Minimum Window Substring
/// Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).
/// 
/// Example:
/// 
/// Input: S = "ADOBECODEBANC", T = "ABC"
/// Output: "BANC"
/// 
/// Note:
/// 
/// If there is no such window in S that covers all characters in T, return the empty string "".
/// If there is such window, you are guaranteed that there will always be only one unique minimum window in S.
/// 
/// 来源：力扣（LeetCode）
/// 链接：https://leetcode-cn.com/problems/minimum-window-substring
/// 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
pub fn min_window(s: String, t: String) -> String {
    if s.is_empty() || s.len() < t.len() {return "".to_string();}
    let cvt = |x: char| {((x as u32) & 0xff) as u8};
    
    const LEN: usize = 256;
    let (mut ac, mut ec) = ([0;LEN], [0; LEN]);
    
    t.chars().for_each(|x| {ec[cvt(x) as usize] += 1;});
    let (mut min_width, mut min_start, mut wnd_start, mut appeared) = (std::i32::MAX, 0, 0, 0);
    
    let ss: Vec<usize> = s.chars().map(|x| {(x as u32) as usize}).collect();
    let mut itr = ss.iter().enumerate();
    
    while let Some(x) = itr.next() {
        // let idx = cvt(x.1) as usize;
        let idx = *x.1;
        if ec[idx] > 0 {
            ac[idx] += 1;
            if ac[idx] <= ec[idx] {
                appeared += 1;
            }
        }
        
        if appeared == t.len() {
            // let mut tmp = s.chars().skip(wnd_start as usize);
            let mut tmp = ss.iter().skip(wnd_start as usize);
            while let Some(y) = tmp.next() {
                // let idx = cvt(y) as usize;
                let idx = *y;
                if (ac[idx] > ec[idx]) || (ec[idx] == 0) {
                    ac[idx] -= 1;
                    wnd_start += 1;
                } else {
                    break;
                }
            }
            
            let tp = x.0 as i32;
            if min_width > (1 + tp - wnd_start) {
                min_width = 1 + tp - wnd_start;
                min_start = wnd_start;
            }
        }
    }
    
    if min_width == std::i32::MAX {"".to_string()}
    else {s.chars().skip(min_start as usize).take(min_width as usize).collect()}
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    
    #[test]
    fn merge() {

        let cases = [
            (vec![vec![2,3],vec![4,5],vec![6,7],vec![8,9],vec![1,10]],vec![vec![1,10]] ),
                (vec![vec![1,4],vec![2,3]], vec![vec![1,4]]),
            (vec![vec![1,3],vec![2,6],vec![8,10],vec![15,18]], vec![vec![1,6],vec![8,10],vec![15,18]]),
            (vec![vec![1,4],vec![4,5]], vec![vec![1,5]]),
        ];

        cases.iter().for_each(|x| {
            assert_eq!(x.1, super::merge(x.0.to_vec()), "cases:{:?}", x.0);
        })
    }

    #[test]
    fn group_anagrams() {

        let cases = [
            (vec!["eat", "tea", "tan", "ate", "nat", "bat"], vec![
                vec!["ate","eat","tea"],
                vec!["nat","tan"],
                vec!["bat"]
            ]),
        ];

        cases.iter().for_each(|x| {
            let mut left = x.1.clone();
            let mut right = super::group_anagrams(x.0.iter().map(|x| {x.to_string()}).collect());
            left.iter_mut().for_each(|tmp| {tmp.sort()});
            right.iter_mut().for_each(|tmp| {tmp.sort()});
            left.sort(); right.sort();
            assert_eq!(left, right, "cases:{:?}", x.0);
        })
    }

    #[test]
    fn multiply() {
        let cases = [
            (("2", "3"), "6"),
            (("123", "456"), "56088"),
        ];
        
        cases.iter().for_each(|x| {
            assert_eq!(x.1, super::multiply((x.0).0.to_string(), (x.0).1.to_string()).as_str(), "cases:{:?}", x.0);
        })
    }
    
    #[test]
    fn find_substring() {
        let cases = [
            (("a", vec!["a"]), vec![0]),
            (("barfoothefoobarman", vec!["foo","bar"]), vec![0, 9]),
            (("wordgoodgoodgoodbestword", vec!["word","good","best","word"]), vec![]),
        ];

        cases.iter().for_each(|x| {
            assert_eq!(x.1, super::find_substring((x.0).0.to_string(), (x.0).1.iter().map(|x|{x.to_string()}).collect()), "case: {:?}", x.0);
        });
    }

    #[test]
    fn divide() {
        let cases = [
            ((-2147483648,-1), 1),
            ((-1,1),-1),
            ((std::i32::MIN,std::i32::MAX),-1),
            ((-10,1), -10),
            ((10,1), 10),
        ];
        
        cases.iter().for_each(|&x| {
            assert_eq!(x.1, super::divide((x.0).0, (x.0).1), "case: {:?}", x.0);
        });
    }

    #[test]
    fn my_pow() {
        let cases = [
            (2.0000f64,10, 1024.0000f64),
            (2.1f64,3,9.261f64),
            (2.0,-2,0.25f64)
        ];
        
        let acc = 0.00001f64;
        for &c in cases.iter() {
            assert_eq!((super::my_pow(c.0, c.1) - c.2).partial_cmp(&acc), Some(Ordering::Less));
        }
    }
    
    #[test]
    fn spiral_order() {
        let cases = [
            (vec![
            ], vec![]),
            (vec![
                vec![ 1, 2, 3 ],
                vec![ 4, 5, 6 ],
                vec![ 7, 8, 9 ],
            ], vec![1,2,3,6,9,8,7,4,5]),
            (vec![
                vec![1, 2, 3, 4],
                vec![5, 6, 7, 8],
                vec![9,10,11,12]
            ], vec![1,2,3,4,8,12,11,10,9,5,6,7]),
            (vec![
                vec![1, 2, 3, 4],
                vec![5, 6, 7, 8],
                vec![9,10,11,12],
                vec![99,90,91,92]
            ], vec![1,2,3,4,8,12,92,91,90,99,9,5,6,7,11,10]),
            (vec![
                vec![1, 2, 3, 4,70],
                vec![5, 6, 7, 8, 71],
                vec![9,10,11,12, 72],
                vec![99,90,91,92,73]
            ], vec![1,2,3,4,70,71,72,73,92,91,90,99,9,5,6,7,8,12,11,10]),
            (vec![
                vec![7],vec![9],vec![6]
            ], vec![7,9,6]),
            (vec![
                vec![1],vec![2],vec![3],vec![4],vec![5],vec![6],vec![7],vec![8],vec![9],vec![10]
            ], vec![1,2,3,4,5,6,7,8,9,10]),
        ];
        
        for c in cases.iter() {
            assert_eq!(super::spiral_order(c.0.clone()), c.1);
        }
    }
    
    #[test]
    fn generate_matrix() {
        let cases = [
            (vec![
                vec![1,2,3],
                vec![8,9,4],
                vec![7,6,5]
            ], 3),
            (vec![
                vec![1]
            ], 1),
            (vec![
                vec![1,2,3,4],
                vec![12,13,14,5],
                vec![11,16,15,6],
                vec![10,9,8,7]
            ], 4),
        ];
        
        for c in cases.iter() {
            assert_eq!(super::generate_matrix(c.1), c.0);
        }
    }
    
    #[test]
    fn my_sqrt() {
        assert_eq!(super::my_sqrt(2147395599), 46339);
    }
    
    #[test]
    fn min_window() {

        let cases = [
            (("a","a"),"a"),
            (("ADOBECODEBANC", "ABC"), "BANC"),
        ];

        cases.iter().for_each(|x| {
            assert_eq!(x.1, super::min_window((x.0).0.to_string(), (x.0).1.to_string()), "cases:{:?}", x.0);
        })
    }
}