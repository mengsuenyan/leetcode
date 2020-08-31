use std::collections::HashSet;

/// 动态规划相关

/// Triangle
/// Given a triangle, find the minimum path sum from top to boom. Each step you may move to adjacent
/// numbers on the row below.
/// For example, given the following triangle
/// [
/// [2],
/// [3,4],
/// [6,5,7],
/// [4,1,8,3]
/// ]
/// e minimum path sum from top to boom is 11 (i.e., 2 + 3 + 5 + 1 = 11).
/// Note: Bonus point if you are able to do this using only O(n) extra space, where n is the total number
/// of rows in the triangle.
pub fn minimum_total(triangle: Vec<Vec<i32>>) -> i32 {
    let mut triangle = triangle;
    if triangle.len() < 2 {
        return match triangle.first() {Some(x) => *(x.first().unwrap()), None => 0,};
    }
    
    for i in (0..=(triangle.len() - 2)).rev() {
        for j in 0..(i+1) {
            triangle[i][j] += std::cmp::min(triangle[i+1][j], triangle[i+1][j+1]);
        }
    }
    
    triangle[0][0]
}

/// Maximum Subarray
/// Find the contiguous subarray within an array (containing at least one number) which has the largest
/// sum.
/// For example, given the array [−2,1,−3,4,−1,2,1,−5,4], the contiguous subarray [4,−1,2,1] has
/// the largest sum = 6.
pub fn max_subarray(a: Vec<i32>) -> i32 {
    match a.first() {
        Some(&x) => {
            a.iter().skip(1).fold((x, x), |(res, f), &ele| {
                let f = std::cmp::max(f+ele, ele);
                (std::cmp::max(f, res), f)
            }).0
        },
        None => 0,
    }
}

/// Palindrome Partitioning II
/// Given a string s, partition s such that every substring of the partition is a palindrome.
/// Return the minimum cuts needed for a palindrome partitioning of s.
/// For example, given s = ”aab”,
/// Return 1 since the palindrome partitioning [”aa”,”b”] could be produced using 1 cut.
pub fn min_cur(s: String) -> i32 {
    if s.len() < 1 {
        return 0;
    }
    
    let s: Vec<char> = s.chars().collect();
    let (mut f, mut p) = (Vec::new(), vec![vec![false;s.len()]; s.len()]);
    (0..=(s.len() as i32)).rev().for_each(|x| {f.push(x-1);});
    
    (0..s.len()).rev().for_each(|i| {
        (i..s.len()).for_each(|j| {
            if s[i]==s[j] && (j - i < 2 || p[i+1][j-1]) {
                p[i][j] = true;
                f[i] = std::cmp::min(f[i], f[j+1]+1);
            }
        });
    });
    
    f[0]
}

/// Maximum Rectangle
/// Given a 2D binary matrix filled with 0’s and 1’s, find the largest rectangle containing all ones and
/// return its area.
pub fn maximal_rectangle(matrix: Vec<Vec<char>>) -> i32 {
    let (_m, n) = match matrix.first() { Some(x) => (matrix.len(), x.len()), None=> return 0};
    
    let (mut h, mut l, mut r) = (vec![0;n], vec![0;n], vec![n;n]);
    let mut res = 0;
    matrix.iter().enumerate().for_each(|(_i, x)| {
        let (mut left, mut right) = (0, n);
        x.iter().enumerate().for_each(|(j, &c)| {
            if c == '1' {
                h[j] += 1;
                l[j] = std::cmp::max(l[j], left);
            } else {
                left = j + 1;
                h[j] = 0; l[j] = 0; r[j] = n;
            }
        });
        x.iter().enumerate().rev().for_each(|(j, &c)| {
            if c == '1' {
                r[j] = std::cmp::min(r[j], right);
                res = std::cmp::max(res, h[j]*(r[j]-l[j]));
            } else {
                right = j;
            }
        });
    });
    
    res as i32
}

/// Best Time to Buy and Sell Stock III
/// Say you have an array for which the i-th element is the price of a given stock on day i.
/// Design an algorithm to find the maximum profit. You may complete at most two transactions.
/// Note: You may not engage in multiple transactions at the same time (ie, you must sell the stock before
/// you buy again).
pub fn max_profit(prices: Vec<i32>) -> i32 {
    let n = if prices.len() < 2 {return 0} else {prices.len()};
    let (mut f, mut g) = (vec![0;n], vec![0;n]);
    
    prices.iter().zip(f.iter_mut()).skip(1).fold((*prices.first().unwrap(), 0), |x, (&price, cur)| {
        let tmp = std::cmp::min(x.0, price);
        *cur = std::cmp::max(x.1, price - tmp);
        (tmp, *cur)
    });
    
    prices.iter().rev().zip(g.iter_mut().rev()).skip(1).fold((*prices.last().unwrap(), 0), |x, (&price, cur)| {
        let tmp = std::cmp::max(x.0, price);
        *cur = std::cmp::max(x.1, tmp - price);
        (tmp, *cur)
    });
    
    f.iter().zip(g.iter()).fold(0, |profit, (&x, &y)| {
        std::cmp::max(profit, x+y)
    })
}

/// Interleaving String
/// Given s1; s2; s3, find whether s3 is formed by the interleaving of s1 and s2.
/// For example, Given: s1 = ”aabcc”, s2 = ”dbbca”,
/// When s3 = ”aadbbcbcac”, return true.
/// When s3 = ”aadbbbaccc”, return false.
pub fn is_interleave(s1: String, s2: String, s3: String) -> bool {
    if (s1.len() + s2.len()) != s3.len() { return false;}
    
    let (s1, s2, s3): (Vec<char>, Vec<char>, Vec<char>) = (s1.chars().collect(), s2.chars().collect(), s3.chars().collect());
    let mut f = vec![vec![true;s2.len()+1];s1.len()+1];
    
    s1.iter().zip(s3.iter()).enumerate().for_each(|(i, (&x, &y))| {
        f[i+1][0] = f[i][0] && (x == y);
    });
    
    s2.iter().zip(s3.iter()).enumerate().for_each(|(i, (&x, &y))| {
        f[0][i+1] = f[0][i] && (x == y);
    });
    
    s1.iter().enumerate().for_each(|(i, &x)| {
        s2.iter().enumerate().for_each(|(j, &y)| {
            f[i+1][j+1] = ((x == s3[i+j+1]) && f[i][j+1]) || ((y == s3[i+j+1]) && f[i+1][j]);
        });
    });
    
    f[s1.len()][s2.len()]
}

/// Scramble String
/// Given a string s1, we may represent it as a binary tree by partitioning it to two non-empty substrings
/// recursively.
/// Below is one possible representation of s1 = ”great”:
/// great
/// / \
/// gr eat
/// / \ / \
/// g r e at
/// / \
/// a t
/// To scramble the string, we may choose any non-leaf node and swap its two children.
/// For example, if we choose the node ”gr” and swap its two children, it produces a scrambled string
/// ”rgeat”.
/// rgeat
/// / \
/// rg eat
/// / \ / \
/// r g e at
/// / \
/// a t
/// We say that ”rgeat” is a scrambled string of ”great”.
/// Similarly, if we continue to swap the children of nodes ”eat” and ”at”, it produces a scrambled string
/// ”rgtae”.
/// rgtae
/// / \
/// rg tae
/// / \ / \
/// r g ta e
/// / \
/// t a
/// We say that ”rgtae” is a scrambled string of ”great”.
/// Given two strings s1 and s2 of the same length, determine if s2 is a scrambled string of s1.
pub fn is_scramble(s1: String, s2: String) -> bool {
    let len = if s1.len() == s2.len() {s1.len()} else {return false;};
    
    let mut f = vec![vec![vec![false; len]; len]; len+1];
    let (s1, s2): (Vec<char>, Vec<char>) = (s1.chars().collect(), s2.chars().collect());
    
    s1.iter().enumerate().for_each(|(i, &x)| {
        s2.iter().enumerate().for_each(|(j, &y)| {
            f[1][i][j] = x == y;
        });
    });
    
    (1..=len).for_each(|n| {
        (0..=(len-n)).for_each(|i| {
            (0..=(len-n)).for_each(|j| {
                for k in 1..n {
                    if (f[k][i][j] && f[n-k][i+k][j+k]) || (f[k][i][j+n-k] && f[n-k][i+k][j]) {
                        f[n][i][j] = true;
                        break;
                    }
                }
            });
        });
    });
    
    f[len][0][0]
}

/// Minimum Path Sum
/// Given a m × n grid filled with non-negative numbers, find a path from top le to boom right which
/// minimizes the sum of all numbers along its path.
/// Note: You can only move either down or right at any point in time
pub fn min_path_sum(grid: Vec<Vec<i32>>) -> i32 {
    let (m, n, first) = match grid.first() {Some(x) => (grid.len(), x.len(), *x.first().unwrap()), None => {return 0;}};
    
    let mut f = vec![vec![0;n];m];
    f[0][0] = first;
    
    (1..m).for_each(|i| {f[i][0] = f[i-1][0] + grid[i][0];});
    (1..n).for_each(|i| {f[0][i] = f[0][i-1] + grid[0][i];});
    
    grid.iter().enumerate().skip(1).for_each(|(i, x)| {
        x.iter().enumerate().skip(1).for_each(|(j, &ele)| {
            f[i][j] = std::cmp::min(f[i-1][j], f[i][j-1]) + ele;
        });
    });
    
    f[m-1][n-1]
}

/// Edit Distance
/// Given two words word1 and word2, find the minimum number of steps required to convert word1 to
/// word2. (each operation is counted as 1 step.)
/// You have the following 3 operations permitted on a word:
/// • Insert a character
/// • Delete a character
/// • Replace a character
pub fn min_distance(word1: String, word2: String) -> i32 {
    let (m, n) = (word1.len(), word2.len());
    let mut f = vec![vec![0; n+1];m+1];
    
    (0..=m).for_each(|i| {f[i][0] = i;});
    (0..=n).for_each(|i| {f[0][i] = i;});
    
    let (word1, word2): (Vec<char>, Vec<char>) = (word1.chars().collect(), word2.chars().collect());
    (1..=m).for_each(|i| {
        (1..=n).for_each(|j| {
            f[i][j] = if word2[j-1] == word1[i-1] {
                f[i-1][j-1]
            } else {
                let tmp = std::cmp::min(f[i-1][j], f[i][j-1]);
                1 + std::cmp::min(tmp, f[i-1][j-1])
            }
        });
    });
    
    f[m][n] as i32
}

/// Decode Ways
/// A message containing leers from A-Z is being encoded to numbers using the following mapping:
/// 'A' -> 1
/// 'B' -> 2
/// ...
/// 'Z' -> 26
/// Given an encoded message containing digits, determine the total number of ways to decode it.
/// For example, Given encoded message "12", it could be decoded as "AB" (1 2) or "L" (12).
/// e number of ways decoding "12" is 2.
pub fn num_decodings(s: String) -> i32 {
    let first = match s.chars().next() {
        Some(x) => {if x == '0' {return 0;} else {x}},
        None => {return 0;},
    };
    
    s.chars().skip(1).fold((1, 1, first), |(prev, cur, pc), c| {
        let cur = if c == '0' {0} else {cur};
        let prev = if !(pc == '1' || (pc == '2' && c <= '6')) {0} else {prev};

        (cur, prev+cur, c)
    }).1
}

/// Distinct Subsequences
/// Given a string S and a string T , count the number of distinct subsequences of T in S.
/// A subsequence of a string is a new string which is formed from the original string by deleting some
/// (can be none) of the characters without disturbing the relative positions of the remaining characters. (ie,
/// "ACE" is a subsequence of "ABCDE" while "AEC" is not).
/// Here is an example: S = "rabbbit", T = "rabbit"
/// Return 3.
pub fn num_distinct(s: String, t: String) -> i32 {
    let mut f = if t.len() <= s.len() {vec![0;t.len()+1]} else {return 0;};
    f[0] = 1;
    
    s.chars().for_each(|sc| {
        t.chars().rev().enumerate().for_each(|(i, tc)| {
            f[t.len() - i] += if sc == tc {f[t.len()-i-1]} else {0}
        });
    });
    
    f[t.len()]
}

/// Word Break
/// Given a string s and a dictionary of words dict, determine if s can be segmented into a space-separated
/// sequence of one or more dictionary words.
/// For example, given
/// s = "leetcode",
/// dict = ["leet", "code"].
/// Return true because "leetcode" can be segmented as "leet code".
pub fn word_break(s: String, dict: Vec<String>) -> bool {
    let mut f = vec![false; s.len()+1];
    f[0] = true;
    
    let mut h = HashSet::with_capacity(dict.len());
    dict.iter().for_each(|x| {h.insert(x.clone());});
    
    let s = s.as_str();
    (1..=s.len()).for_each(|i| {
        for j in (0..i).rev() {
            if f[j] && h.contains(&s[j..i]) {
                f[i] = true;
                break;
            }
        }
    });
    
    f[s.len()]
}

/// Word Break II
/// Given a string s and a dictionary of words dict, add spaces in s to construct a sentence where each
/// word is a valid dictionary word.
/// Return all such possible sentences.
/// For example, given
/// s = "catsanddog",
/// dict = ["cat", "cats", "and", "sand", "dog"].
/// A solution is ["cats and dog", "cat sand dog"].
pub fn word_break_ii(s: String, dict: Vec<String>) -> Vec<String> {
    struct F(Vec<(usize, usize)>, Vec<String>, fn(&str, &Vec<Vec<bool>>, usize, &mut F));
    let dfs = |s: &str, prev: &Vec<Vec<bool>>, cur: usize, f: &mut F| {
        if cur == 0 {
            let mut tmp = String::with_capacity(s.len() * (f.0).len());
            (f.0).iter().rev().for_each(|&x| {
                tmp.push_str(&s[x.0..x.1]);
                tmp.push(' ');
            });
            tmp.pop();
            (f.1).push(tmp);
        }
        
        prev[cur].iter().enumerate().for_each(|(i, &is_exist)| {
            if is_exist {
                (f.0).push((i, cur));
                (f.2)(s, prev, i, f);
                (f.0).pop();
            }
        })
    };
    
    let mut f = vec![false; s.len() + 1];
    let mut prev = vec![vec![false;s.len()];s.len()+1];
    f[0] = true;

    let s = s.as_str();
    let mut h = HashSet::with_capacity(dict.len());
    dict.iter().for_each(|x| {h.insert(x.clone());});
    (1..=s.len()).for_each(|i| {
        for j in (0..i).rev() {
            if f[j] && h.contains(&s[j..i]) {
                f[i] = true;
                prev[i][j] = true;
            }
        }
    });
    
    let mut f = F(Vec::new(), Vec::new(), dfs);
    
    dfs(s, &prev, s.len(), &mut f);
    
    f.1
}
