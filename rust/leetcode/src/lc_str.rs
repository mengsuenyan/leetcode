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
}