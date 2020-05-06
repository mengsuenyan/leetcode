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
}