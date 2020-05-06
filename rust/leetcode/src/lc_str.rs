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
}