use std::str::Chars;
use std::iter::Peekable;

pub fn longest_common_prefix(strs: Vec<String>) -> String {
    if strs.is_empty() {
        return String::from("");
    }
    
    let mut itr = strs.iter();
    let mut first = itr.next().unwrap().clone();

    for s in itr {
        for _ in 0..first.len() {
            if s.starts_with(first.as_str()) {
                break;
            } else {
                first.pop();
            }
        }
    }
    
    return first
}

/// Given two binary strings, return their sum (also a binary string).  
///   
/// The input strings are both non-empty and contains only characters 1 or 0.  
///   
/// 来源：力扣（LeetCode）  
/// 链接：https://leetcode-cn.com/problems/add-binary  
/// 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。  
pub fn add_binary(a: String, b: String) -> String {
    let (aitr, mut bitr) = if a.len() < b.len() {
        (b.as_bytes().iter().rev(), a.as_bytes().iter().rev())
    } else {
        (a.as_bytes().iter().rev(), b.as_bytes().iter().rev())
    };
    let mut re = Vec::with_capacity(a.len() + b.len() + 1);
    let mut carry = false;
    
    for &ea in aitr {
        let (cry, ele) = match bitr.next() {
            Some(&eb) => {
                match (ea, eb) {
                    (b'0', b'0') => (false, if carry {b'1'} else {b'0'}),
                    (b'0', b'1') => if carry {(true, b'0')} else {(false, b'1')},
                    (b'1', b'0') => if carry {(true, b'0')} else {(false, b'1')},
                    (b'1', b'1') => if carry {(true, b'1')} else {(true, b'0')},
                    _ => {unreachable!()},
                }
            },
            None => {
                match ea {
                    b'0' => (false, if carry {b'1'} else {b'0'}),
                    b'1' => (if carry {(true, b'0')} else {(false, b'1')}),
                    _ => {unreachable!()},
                }
            },
        };
        carry = cry;
        re.push(ele);
    }

    if carry {
        re.push(b'1');
    }
    
    re.reverse();
    String::from_utf8(re).unwrap()
}

/// The count-and-say sequence is the sequence of integers with the first five terms as following:  
///   
/// 1 is read off as "one 1" or 11.  
/// 11 is read off as "two 1s" or 21.  
/// 21 is read off as "one 2, then one 1" or 1211.  
///   
/// Given an integer n where 1 ≤ n ≤ 30, generate the nth term of the count-and-say sequence. You can do so recursively, in other words from the previous member read off the digits, counting the number of digits in groups of the same digit.  
///   
/// Note: Each term of the sequence of integers will be represented as a string.  
/// O(n!)
pub fn count_and_say(n: i32) -> String {
    let mut v = vec![b'1'];

    for _ in 1..n {
        let mut subv = Vec::new();

        let mut tgt = v.first().unwrap().clone();
        let mut cnt = 0;
        for &ele in v.iter() {
            if ele == tgt {
                cnt += 1;
            } else {
                subv.push(b'0' + cnt);
                subv.push(tgt);
                tgt = ele;
                cnt = 1;
            }
        }

        if cnt > 0 {
            subv.push(b'0' + cnt);
            subv.push(tgt);
        }
        
        v.clear();
        v.append(&mut subv);
    }

    String::from_utf8(v).unwrap()
}

/// Implement strStr().  
///   
/// Return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.  
///   
/// 来源：力扣（LeetCode）  
/// 链接：https://leetcode-cn.com/problems/implement-strstr  
/// 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。  
/// O(n)  
pub fn str_str(haystack: String, needle: String) -> i32 {
    if haystack.is_empty() && needle.is_empty() {
        return 0;
    }
    
    let hs = haystack.as_str();
    let nd = needle.as_str();
    for i in 0..hs.len() {
        let sub = &hs[i..];
        if sub.starts_with(nd) {
            return i as i32;
        }
    }
    
    -1
}

/// Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.  
///   
/// An input string is valid if:  
///   
/// Open brackets must be closed by the same type of brackets.  
/// Open brackets must be closed in the correct order.  
///   
/// Note that an empty string is also considered valid.  
///   
/// 来源：力扣（LeetCode）  
/// 链接：https://leetcode-cn.com/problems/valid-parentheses  
/// 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。  
/// O(n)
pub fn is_valid(s: String) -> bool {
    if s.is_empty() {
        return true;
    }
    
    let mut want = Vec::with_capacity(s.len()/2);
    for &ele in s.as_bytes().iter() {
        if ele == b'(' {
            want.push(b')');
        } else if ele == b'{' {
            want.push(b'}');
        } else if ele == b'[' {
            want.push(b']');
        } else {
            match want.pop() {
                Some(w) => if w != ele { return false; },
                None => { return false; },
            }
        }
    }

    want.is_empty()
}

/// Valid Palindrome
/// 
/// Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring
/// cases.
/// For example,
/// ”A man, a plan, a canal: Panama” is a palindrome.
/// ”race a car” is not a palindrome.
/// Note: Have you consider that the string might be empty? is is a good question to ask during an
/// interview.
/// For the purpose of this problem, we define empty string as valid palindrome.
pub fn is_palindrome(s: String) -> bool {
    let (mut itr, mut ritr) = (s.chars(), s.chars().rev());
    let check_next = |x: &mut std::iter::Rev<Chars>| {
        loop {
            match x.next() {
                Some(x) => {
                    if x.is_ascii_digit() { return Some(x); }
                    else if x.is_ascii_alphabetic() {return Some(x.to_ascii_lowercase());}
                },
                None => {return None;},
            }
        }
    };
    
    loop {
        match itr.next() {
            Some(x) => {
                if x.is_ascii_alphanumeric() {
                    match check_next(&mut ritr) {
                        Some(y) => {
                            if x.to_ascii_lowercase() != y {return false;}
                        },
                        None => {return false;}
                    }
                }
            },
            None => {
                return true;
            }
        };
    }
}

/// Longest Palindromic Substring
/// 
/// Given a string S, find the longest palindromic substring in S. You may assume that the maximum
/// length of S is 1000, and there exists one unique longest palindromic substring.
pub fn longest_palindrome(s: String) -> String {
    // Manacher's algorithm
    // 插入了特殊标记#后, 回文个数必然是奇数的. 以某一位置对称的的半轴长必然是原始回文的长度;
    let mut ps = Vec::with_capacity((s.len() << 1) + 3);
    ps.push('^');
    s.chars().for_each(|x| {ps.push('#'); ps.push(x);});
    ps.push('#');
    ps.push('$');
    
    // cnt记录以i位置对称的长度(不包括自身), c记录上一次最长对称的中心位置, r记录遍历过的最远位置
    let (mut cnt, mut c, mut r) = (Vec::new(), 0usize, 0);
    cnt.resize(ps.len(), 0);
    for i in 1..(ps.len()-1) {
        // 关于c与i对称的位置(c - (i-c))
        let m = (c << 1).wrapping_sub(i);
        // 跳过已经比较过对称的元素, 因为m关于c和i对称, 如果r大于i, 那么m实在以c为中心的对称的轴上的,
        // 那么m对称的轴和以i为对称的轴必然有重叠, 重叠便是min(r-i,cnt[m])
        // r'----xx-m-xx----c--xx-i-xx----r
        cnt[i] = if r > i {std::cmp::min(r - i, cnt[m])} else {0};
        
        // 以T[i]为中心, 向左右两边查找对称
        while ps[i + 1 + cnt[i]] == ps[i - 1 - cnt[i]] {
            cnt[i] += 1;
        }
        
        if i + cnt[i] > r {
            c = i;
            r = i + cnt[i];
        }
    }
    
    match cnt.iter().enumerate().max_by(|&x, &y| {
        x.1.cmp(y.1)
    }) {
        Some((center_idx, &max_len)) => {
            s.chars().skip((center_idx - 1 - max_len) >> 1).take(max_len).collect()
        },
        None => String::new(),
    }
}

/// Regular Expression Mating
///
/// Implement regular expression matching with support for '.' and '*'.
/// '.' Matches any single character. '*' Matches zero or more of the preceding element.
/// e matching should cover the entire input string (not partial).
/// e function prototype should be:
/// bool isMatch(const char *s, const char *p)
/// Some examples:
/// isMatch("aa","a") → false
/// isMatch("aa","aa") → true
/// isMatch("aaa","aa") → false
/// isMatch("aa", "a*") → true
/// isMatch("aa", ".*") → true
/// isMatch("ab", ".*") → true
/// isMatch("aab", "c*a*b") → true
pub fn is_match(s: String, p: String) -> bool {
    
    is_match_help_(s.chars().peekable(), p.chars().peekable())
}

fn is_match_help_(mut s: Peekable<Chars>, mut p: Peekable<Chars>) -> bool{
    if p.peek().is_none() { return s.peek().is_none();}

    let mut pn = p.clone();
    pn.next();
    match pn.peek() {
        Some(&'*') => {
            let mut pnn = pn.clone();
            pnn.next();

            // 检查'*'匹配多个
            while (p.peek() == s.peek()) || (p.peek() == Some(&'.') && s.peek().is_some()) {
                // 检查'*'匹配0个
                if is_match_help_(s.clone(), pnn.clone()) {
                    return true;
                }
                s.next();
            }
            is_match_help_(s.clone(), pnn)
        },
        _ => {
            // 字符匹配
            if (p.peek() == s.peek()) || (p.peek() == Some(&'.') && s.peek().is_some()) {
                s.next();
                is_match_help_(s.clone(), pn.clone())
            } else {
                false
            }
        },
    }
}

/// Wildcard Mating
/// 
/// Implement wildcard paern matching with support for '?' and '*'.
/// '?' Matches any single character. '*' Matches any sequence of characters (including the empty
/// sequence).
/// e matching should cover the entire input string (not partial).
/// e function prototype should be:
/// bool isMatch(const char *s, const char *p)
/// Some examples:
/// isMatch("aa","a") → false
/// isMatch("aa","aa") → true
/// isMatch("aaa","aa") → false
/// isMatch("aa", "*") → true
/// isMatch("aa", "a*") → true
/// isMatch("ab", "?*") → true
/// isMatch("aab", "c*a*b") → false
pub fn is_wildcard_match(s: String, p: String) -> bool {
    let (mut sitr, mut pitr) 
        = (s.chars().enumerate().skip(0).peekable(), p.chars().enumerate().skip(0).peekable());
    let mut star = false;
    let (mut ss, mut ps) = (0, 0);
    
    while sitr.peek().is_some() {
        match pitr.peek() {
            Some(&(_, '?')) => {
                sitr.next();
                pitr.next();
            },
            Some(&(_, '*')) => {
                star = true;
                while let Some(&(_, '*')) = pitr.peek() { pitr.next(); }
                if pitr.peek().is_none() { return true;}
                ss = sitr.peek().unwrap().0;
                ps = pitr.peek().unwrap().0;
            },
            _ => {
                if pitr.peek().is_none() || sitr.peek().unwrap().1 != pitr.peek().unwrap().1 {
                    if !star {return false;}
                    ss += 1;
                    sitr = s.chars().enumerate().skip(ss).peekable();
                    pitr = p.chars().enumerate().skip(ps).peekable();
                } else {
                    sitr.next();
                    pitr.next();
                }
            }
        }
    }

    while let Some(&(_, '*')) = pitr.peek() {pitr.next();}
    pitr.peek().is_none()
}


/// Valid Number
/// 
/// Validate if a given string is numeric.
/// Some examples:
/// "0" => true
/// " 0.1 " => true
/// "abc" => false
/// "1 a" => false
/// "2e10" => true
/// Note: It is intended for the problem statement to be ambiguous. You should gather all requirements
/// up front before implementing one
pub fn is_number(s: String) -> bool {
    // invalid, space, sign, digit, dot, exponent
    let ct = [0, 1, 2, 3, 4, 5];
    let tb = [
        //i  sp si di do  e
        [-1, 0, 3, 1, 2, -1], /*initial state*/
        [-1, 8,-1, 1, 4, 5],  /*digit*/
        [-1, -1,-1,4,-1,-1],/*dot is occur*/
        [-1, -1,-1,1, 2,-1],/*sign is occur*/
        [-1, 8, -1,4,-1, 5],/*float*/
        [-1, -1, 6,7,-1,-1],/*exponent is occur*/
        [-1, -1,-1,7,-1,-1],/*sign is occur after exponent*/
        [-1, 8, -1,7,-1,-1],/*efloat*/
        [-1, 8, -1,-1,-1,-1],/*final state*/
    ];
    
    let mut st = 0;
    for ele in s.chars() {
        let t = if ele == ' ' {
            ct[1]
        } else if ele.is_ascii_digit() {
            ct[3]
        } else if (ele == '+') || (ele == '-') {
            ct[2]
        } else if (ele == 'e') || (ele == 'E') {
            ct[5]
        } else if ele == '.' {
            ct[4]
        } else {
            ct[0]
        };
        
        st = tb[st as usize][t];
        if st == -1 {
            break;
        }
    }
    
    st == 1 || st == 4 || st == 7 || st == 8
}

/// Integer to Roman
/// 
/// Given an integer, convert it to a roman numeral.
/// Input is guaranteed to be within the range from 1 to 3999.
pub fn int_to_roman(num: i32) -> String {
    debug_assert!(num >0 && num < 4000, "num must be belong to (0,4000)");
    let (mut num, mut i) = (num, 0);
    let radix = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1];
    let sym = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX",
        "V", "IV", "I"];
    
    let mut roman = String::with_capacity(128);
    while num > 0 {
        let cnt = num / radix[i];
        num = num % radix[i];
        (1..=cnt).rev().for_each(|_| { roman.push_str(sym[i])});
        i += 1;
    }
    
    roman
}

pub fn roman_to_int(roman: String) -> i32 {
    let map = |x: char| {
        match x {
            'I' => 1,
            'V' => 5,
            'X' => 10,
            'L' => 50,
            'C' => 100,
            'D' => 500,
            'M' => 1000,
            _ => 0,
        }
    };
    
    if roman.is_empty() {
        0
    } else {
        let mut res = 0;
        
        roman.chars().take(1).for_each(|x| {res += map(x);});
        roman.chars().skip(1).zip(roman.chars().take(roman.len() - 1)).for_each(|(x, y)| {
            res += if map(x) > map(y) {
                map(x) - (map(y) << 1)
            } else {
                map(x)
            };
        });

        res
    }
}

/// Anagrams
/// 
/// Given an array of strings, return all groups of strings that are anagrams.
/// Note: All inputs will be in lower-case.
pub fn anagrams(strs: Vec<String>) -> Vec<String> {
    let mut group = std::collections::HashMap::with_capacity(strs.len() >> 1);
    
    strs.iter().for_each(|x| {
        let mut tmp = Vec::with_capacity(x.len());
        x.chars().for_each(|c| {
            if c.is_ascii_lowercase() {
                tmp.push(c as u8);
            }
        });
        tmp.sort();
        group.entry(tmp).or_insert(Vec::new()).push(x);
    });
    
    let mut res = Vec::new();
    group.iter().for_each(|x| {
        if x.1.len() > 1 {
            x.1.iter().for_each(|&y| {
                res.push(y.to_string());
            });
        }
    });
    
    res
}

/// Simplify Path
/// 
/// Given an absolute path for a file (Unix-style), simplify it.
/// For example,
/// path = "/home/", => "/home"
/// path = "/a/./b/../../c/", => "/c"
/// Corner Cases:
/// • Did you consider the case where path = "/../"? In this case, you should return "/".
/// • Another corner case is the path might contain multiple slashes '/' together, such as
/// "/home//foo/". In this case, you should ignore redundant slashes and return "/home/foo"
pub fn simplify_path(path: String) -> String {
    let mut dirs = Vec::new();
    
    let mut itr = path.chars().enumerate().peekable();
    
    while itr.peek().is_some() {
        itr.next();
        
        let i = if itr.peek().is_none() {break;} else {itr.peek().unwrap().0};
        
        while let Some(&x) = itr.peek() {
            if x.1 == '/' {
                break;
            }
            itr.next();
        }
        
        let j = if itr.peek().is_none() {path.len()} else {itr.peek().unwrap().0};
        let dir: String = path.chars().skip(i).take(j - i).collect();
        if !dir.is_empty() && dir != "." {
            if dir == ".." {
                dirs.pop();
            } else {
                dirs.push(dir);
            }
        }
    }
    
    if dirs.is_empty() {
        String::from("/")
    } else {
        let mut out = String::new();
        dirs.iter().for_each(|x| {
            out.push('/');
            out.push_str(x.as_str());
        });
        out
    }
}

/// Length of Last Word
/// 
/// Given a string s consists of upper/lower-case alphabets and empty space characters ' ', return the
/// length of last word in the string.
/// If the last word does not exist, return 0.
/// Note: A word is defined as a character sequence consists of non-space characters only.
/// For example, Given s = "Hello World", return 5.
pub fn length_of_last_word(s: String) -> i32 {
    let mut res = 0;
    
    let s = s.trim_end();
    for ele in s.chars().rev() {
        if ele == ' ' {
            break;
        } else {
            res += 1;
        }
    }
    
    res
}

#[cfg(test)]
mod tests {

    #[test]
    fn longest_common_prefix() {
        let cases = [
            (vec![String::from("flower"), String::from("flow"), String::from("flight")], String::from("fl")),
            (vec![String::from("dog"), String::from("racecar"), String::from("car")], String::from("")),
            (vec![], String::from("")),
            (vec![String::from("test")], String::from("test")),
        ];
        
        for c in cases.iter() {
            assert_eq!(super::longest_common_prefix(c.0.clone()), c.1);
        }
    }
    
    #[test]
    fn add_binary() {
        let cases = [
            ("11", "1", "100"),
        ];
        for c in cases.iter() {
            assert_eq!(super::add_binary(String::from(c.0), String::from(c.1)), String::from(c.2));
        }
    }
    
    #[test]
    fn count_and_say() {
        let cases = [
            (1, "1"),
            (2, "11"),
            (3, "21"),
            (4, "1211"),
            (5, "111221"),
        ];
        
        for c in cases.iter() {
            assert_eq!(super::count_and_say(c.0).as_str(), c.1);
        }
    }
    
    #[test]
    fn is_palindrome() {

        let cases = [
            ("0P", false),
            ("A man, a plan, a canal: Panama", true),
            ("race a car", false),
        ];

        cases.iter().for_each(|x| {
            assert_eq!(x.1, super::is_palindrome(String::from(x.0)), "cases: {:?}", x.0);
        });
    }
    
    #[test]
    fn longest_palindrome() {

        let cases = [
            // ("aacdefcaa", "aa"),
            ("bxabad", "aba"),
            // ("cbbd", "bb"),
        ];

        cases.iter().for_each(|&x| {
            assert_eq!(x.1, super::longest_palindrome(String::from(x.0)), "cases: {:?}", x.0);
        });
    }
    
    #[test]
    fn is_match() {
        let cases = [
            (("aa", "a*"), true),
            (("aa","a"), false),
            (("aa", "aa"), true),
            (("aaa", "aa"), false),
            (("aa", ".*"), true),
            (("ab", ".*"), true),
            (("aab", "c*a*b"), true),
        ];
        cases.iter().for_each(|&x| {
            assert_eq!(x.1, super::is_match((x.0).0.to_string(), (x.0).1.to_string()), "cases: {:?}", x.0);
        });
    }
    
    #[test]
    fn is_wildcard_match() {

        let cases = [
            (("adceb", "*a*b"), true),
            (("acdcb","a*c?b"), false),
            (("", "?"), false),
            (("aa","a"), false),
            (("aa", "aa"), true),
            (("aaa", "aa"), false),
            (("aa", "*"), true),
            (("aa", "a*"), true),
            (("ab", "?*"), true),
            (("aab", "c*a*b"), false),
            (("abbabaaabbabbaababbabbbbbabbbabbbabaaaaababababbbabababaabbababaabbbbbbaaaabababbbaabbbbaabbbbababababbaabbaababaabbbababababbbbaaabbbbbabaaaabbababbbbaababaabbababbbbbababbbabaaaaaaaabbbbbaabaaababaaaabb",
              "**aa*****ba*a*bb**aa*ab****a*aaaaaa***a*aaaa**bbabb*b*b**aaaaaaaaa*a********ba*bbb***a*ba*bb*bb**a*b*bb"), false),
        ];
        cases.iter().for_each(|&x| {
            assert_eq!(x.1, super::is_wildcard_match((x.0).0.to_string(), (x.0).1.to_string()), "cases: {:?}", x.0);
        });
    }
    
    #[test]
    fn is_number() {

        let cases = [
            ("+.8", true),
            ("0", true),
            (" 0.1 ", true),
            ("abc", false),
            ("1 a", false),
            ("2e10", true),
        ];
        cases.iter().for_each(|&x| {
            assert_eq!(x.1, super::is_number(x.0.to_string()), "cases: {:?}", x.0);
        });
    }
    
    #[test]
    fn simplify_path() {

        let cases = [
            ("/home/", "/home"),
            ("/a/./b/../../c/", "/c"),
            ("/../", "/"),
            ("/home//foo", ("/home/foo")),
        ];
        cases.iter().for_each(|&x| {
            assert_eq!(x.1, super::simplify_path(x.0.to_string()).as_str(), "cases: {:?}", x.0);
        });
    }
    
    #[test]
    fn length_of_last_word() {

        let cases = [
            ("Hello World", 5),
        ];
        cases.iter().for_each(|&x| {
            assert_eq!(x.1, super::length_of_last_word(x.0.to_string()), "cases: {:?}", x.0);
        });
    }
}