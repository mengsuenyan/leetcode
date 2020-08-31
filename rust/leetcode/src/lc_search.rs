//! 搜索相关

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hasher};
use std::collections::HashMap;

/// Search for a Range
/// 
/// Given a sorted array of integers, find the starting and ending position of a given target value.
/// Your algorithm’s runtime complexity must be in the order of O(log n).
/// If the target is not found in the array, return [-1, -1].
/// For example, Given [5, 7, 7, 8, 8, 10] and target value 8, return [3, 4]
pub fn search_range(a: Vec<i32>, tgt: i32) -> Vec<i32> {
    match a.binary_search(&tgt) {
        Ok(x) => {
            let (mut i, mut j) = (x as i32, x as i32);
            for &ele in a.iter().take(x).rev() {
                if ele == tgt {
                    i -= 1;
                } else {
                    break;
                }
            }
            for &ele in a.iter().skip(x+1) {
                if ele == tgt {
                    j += 1;
                } else {
                    break;
                }
            }
            vec![i, j]
        },
        Err(_) => {
            vec![-1,-1]
        }
    }
}

/// Search a 2D Matrix
///
/// Write an efficient algorithm that searches for a value in an m×n matrix. This matrix has the following
/// properties:
/// - Integers in each row are sorted from left to right.
/// - Thee first integer of each row is greater than the last integer of the previous row.
/// For example, Consider the following matrix:
/// ```Rust
/// vec![
///     vec![1, 3, 5, 7],
///     vec![10, 11, 16, 20],
///     vec![23, 30, 34, 50]
/// ]
/// ```
/// Given target = 3, return true
pub fn search_matrix(matrix: Vec<Vec<i32>>, target: i32) -> bool {
    if matrix.is_empty() || matrix.first().unwrap().is_empty() {
        return false;
    }
    
    match matrix.binary_search_by(|x| {
        if &target < x.first().unwrap() {
            std::cmp::Ordering::Greater
        } else if &target > x.last().unwrap() {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Equal
        }
    }) {
        Ok(x) => {
            matrix[x].binary_search(&target).is_ok()
        },
        _ => false,
    }
}

/// Word Ladder
/// Given two words (start and end), and a dictionary, find the length of shortest transformation sequence
/// from start to end, such that:
/// • Only one letter can be changed at a time
/// • Each intermediate word must exist in the dictionary
/// For example, Given:
/// start = "hit"
/// end = "cog"
/// dict = ["hot","dot","dog","lot","log"]
/// As one shortest transformation is ”hit” -> ”hot” -> ”dot” -> ”dog” -> ”cog”, return its length 5.
/// Note:
/// • Return 0 if there is no such transformation sequence.
/// • All words have the same length.
/// • All words contain only lowercase alphabetic characters.
pub fn ladder_length(start: String, end: String, dict: Vec<String>) -> i32 {
    if start.len() != end.len() { return 0; }
    if start.is_empty() || end.is_empty() {return 0; }

    let (mut hs, mut he) = (DefaultHasher::new(), DefaultHasher::new());
    let start_c: Vec<u32> = start.chars().map(|x| {x as u32}).collect();
    start_c.iter().for_each(|&x| {hs.write_u32(x);});
    end.chars().for_each(|x| {he.write_u32(x as u32);});
    let (start, end) = (hs.finish(), he.finish());

    let (mut next_r, mut cur_r) = (std::collections::HashSet::new(), std::collections::HashSet::new());
    let (mut next, mut cur) = (&mut next_r, &mut cur_r);
    let mut visited = std::collections::HashSet::new();
    let (mut level, mut is_found) = (0, false);
    let (mut dicts, mut az) = (std::collections::HashMap::new(), std::collections::HashSet::new());
    dict.iter().for_each(|x| {
        let mut h = DefaultHasher::new();
        let tmp: Vec<u32> = x.chars().map(|x| {
            az.insert(x as u32);
            h.write_u32(x as u32);
            x as u32
        }).collect();

        dicts.insert(h.finish(), tmp);
    });

    if !dicts.contains_key(&end) { return 0;}

    cur.insert(start);

    while !cur.is_empty()  && !is_found {
        level += 1;
        'out: for &key in cur.iter() {
            let word = dicts.get(&key).unwrap_or(&start_c);
            for x in word.iter().enumerate() {
                let mut h = DefaultHasher::new();
                for c in az.iter() {
                    if c == x.1 {continue;} else {
                        word.iter().take(x.0).for_each(|&tmp| {h.write_u32(tmp);});
                        h.write_u32(*c);
                    }
                    word.iter().skip(x.0+1).for_each(|&tmp| {h.write_u32(tmp);});
                    let hv = h.finish();
                    if hv == end {
                        is_found = true;
                        break 'out;
                    }

                    if dicts.contains_key(&hv) && !visited.contains(&hv) {
                        next.insert(hv);
                        visited.insert(hv);
                    }
                    h = DefaultHasher::new();
                }
            }
        }
        cur.clear();
        std::mem::swap(&mut cur, &mut next);
    }


    if is_found {level + 1}
    else {0}
}

pub fn find_ladders(start: String, end: String, dict: Vec<String>) -> Vec<Vec<String>> {
    if start.len() != end.len() { return Vec::new(); }
    if start.is_empty() || end.is_empty() {return Vec::new(); }

    let (mut hs, mut he) = (DefaultHasher::new(), DefaultHasher::new());
    let start_c: Vec<u32> = start.chars().map(|x| {x as u32}).collect();
    start_c.iter().for_each(|&x| {hs.write_u32(x);});
    end.chars().for_each(|x| {he.write_u32(x as u32);});
    let (start, end) = (hs.finish(), he.finish());
    let mut father = std::collections::HashMap::new();

    let (mut next_r, mut cur_r) = (std::collections::HashSet::new(), std::collections::HashSet::new());
    let (mut next, mut cur) = (&mut next_r, &mut cur_r);
    let mut visited = std::collections::HashSet::new();
    let mut is_found = false;
    let (mut dicts, mut az) = (std::collections::HashMap::new(), std::collections::HashSet::new());
    dict.iter().for_each(|x| {
        let mut h = DefaultHasher::new();
        let tmp: Vec<u32> = x.chars().map(|x| {
            az.insert(x as u32);
            h.write_u32(x as u32);
            x as u32
        }).collect();

        dicts.insert(h.finish(), tmp);
    });

    if !dicts.contains_key(&end) { return Vec::new();}

    dicts.insert(start, start_c.clone());
    cur.insert(start);

    while !cur.is_empty()  && !is_found {
        
        cur.iter().for_each(|&x| {visited.insert(x);});
        
        for &key in cur.iter() {
            let word = dicts.get(&key).unwrap_or(&start_c);
            for x in word.iter().enumerate() {
                let mut h = DefaultHasher::new();
                for c in az.iter() {
                    if c == x.1 {continue;} else {
                        word.iter().take(x.0).for_each(|&tmp| {h.write_u32(tmp);});
                        h.write_u32(*c);
                    }
                    word.iter().skip(x.0+1).for_each(|&tmp| {h.write_u32(tmp);});
                    let hv = h.finish();
                    if hv == end {
                        is_found = true;
                    }

                    if !visited.contains(&hv) && (dicts.contains_key(&hv) || hv == end) {
                        next.insert(hv);
                        father.entry(hv).or_insert(Vec::new()).push(key);
                    }
                    h = DefaultHasher::new();
                }
            }
        }
        cur.clear();
        std::mem::swap(&mut cur, &mut next);
    }

    let mut res = Vec::new();
    if is_found {
        let mut path = Vec::new();
        build_path(&mut res, &mut path, start, end, &father, &dicts);
    }
    
    res
}

fn build_path(res: &mut Vec<Vec<String>>, path: &mut Vec<String>, start: u64, word: u64, father: &HashMap<u64, Vec<u64>>, 
              dict: &HashMap<u64, Vec<u32>>) {
    let x: String = dict.get(&word).unwrap().iter().map(|&x| {(x as u8) as char}).collect();
    path.push(x);
    if word == start {
        let mut tmp = path.clone();
        tmp.reverse();
        res.push(tmp);
    } else {
        for &f in father.get(&word).unwrap().iter() {
            build_path(res, path, start, f, father, dict);
        }
    }
    path.pop();
}

/// Surrounded Regions
/// Given a 2D board containing 'X' and 'O', capture all regions surrounded by 'X'.
/// A region is captured by flipping all 'O's into 'X's in that surrounded region .
/// For example,
/// X X X X
/// X O O X
/// X X O X
/// X O X X
/// Aer running your function, the board should be:
/// X X X X
/// X X X X
/// X X X X
/// X O X X
pub fn surrounded_regions(board: &mut Vec<Vec<char>>) {
    let bfs = |board: &mut Vec<Vec<char>>, cor: (usize, usize), size: (usize, usize)| {
        let visit = |board: &mut Vec<Vec<char>>, cor: (usize, usize), size: (usize, usize), q: &mut Vec<(usize, usize)>| {
            if cor.0 >= size.0 || cor.1 >= size.1 || board[cor.0][cor.1] != 'O' {
                return;
            }
            board[cor.0][cor.1] = '+';
            q.push((cor.0, cor.1));
        };

        let mut q = Vec::with_capacity(size.0 * size.1);
        visit(board, cor, size, &mut q);
        while let Some((i, j)) = q.pop() {
            if i > 1 {
                visit(board, (i-1, j), size, &mut q);
            }
            if j > 1 {
                visit(board, (i, j - 1), size, &mut q);
            }
            visit(board, (i+1,j), size, &mut q);
            visit(board, (i, j+1), size, &mut q);
        }
    };

    if board.is_empty() || board[0].is_empty() {
        return;
    }

    let size = (board.len(), board[0].len());
    (0..size.1).for_each(|j| {
        bfs(board, (0, j), size);
        bfs(board, (size.0-1, j), size);
    });
    (1..(size.0-1)).for_each(|i| {
        bfs(board, (i, 0), size);
        bfs(board, (i, size.1-1), size);
    });

    board.iter_mut().for_each(|x| {
        x.iter_mut().for_each(|y| {
            if *y == 'O' {
                *y = 'X';
            } else if *y == '+' {
                *y = 'O';
            }
        });
    });
}

/// Palindrome Partitioning
/// Given a string s, partition s such that every substring of the partition is a palindrome.
/// Return all possible palindrome partitioning of s.
/// For example, given s = ”aab”, Return
/// [
/// ["aa","b"],
/// ["a","a","b"]
/// ]
pub fn partition(s: String) -> Vec<Vec<String>> {
    let mut res = Vec::new();
    let mut output = Vec::new();

    let s: Vec<u8> = s.chars().map(|x| {(x as u32) as u8}).collect();
    palindrome_partition_dfs(&s, 0, &mut output, &mut res);

    res
}

fn palindrome_partition_dfs<'a>(s: &'a Vec<u8>, start: usize, output: &mut Vec<&'a [u8]>, res: &mut Vec<Vec<String>>) {
    if start == s.len() {
        res.push(output.iter().map(|&x| {
            x.iter().map(|&y| {
                y as char
            }).collect()
        }).collect());
        return;
    }

    for i in start..s.len() {
        // 以s[start,i]为树头, 开始查找后续是否是回文, 共s.len()颗树
        if is_palindrome(s, start, i) {
            if let Some(x) = s.get(start..=i) {
                output.push(x);
            }
            palindrome_partition_dfs(s, i+1, output, res);
            output.pop();
        }
    }
}

fn is_palindrome(s: &Vec<u8>, start: usize, end: usize) -> bool {
    let mut itr = s.iter().enumerate().skip(start)
        .zip(s.iter().enumerate().skip(start).take(1+end.saturating_sub(start)).rev());
    while let Some(x) = itr.next() {
        if (x.0).0 < (x.1).0 {
            if (x.0).1 != (x.1).1 {
                return false;
            }
        } else {
            break;
        }
    }

    true
}

/// Unique Paths
/// A robot is located at the top-left corner of a m × n grid (marked ’Start’ in the diagram below).
/// e robot can only move either down or right at any point in time. e robot is trying to reach the
/// boom-right corner of the grid (marked ’Finish’ in the diagram below).
/// How many possible unique paths are there?
/// Note: m and b will be at most 100.
pub fn unique_paths(m: i32, n: i32) -> i32 {
    // f[i][j] = f[i-1][j]+f[i][j-1]
    let (min, max) = if m < n {(m as usize, n as usize)} else {(n as usize, m as usize)};
    let mut f = Vec::with_capacity(min);
    f.resize(min, 0);
    
    match min {
        1 => 1,
        _ => (0..max).fold(0, |_, _| {
        f.iter_mut().skip(1).fold(1, |x, y| {*y = x + *y; *y})})
    }
}

/// Unique Paths II
/// Follow up for ”Unique Paths”:
/// Now consider if some obstacles are added to the grids. How many unique paths would there be?
/// An obstacle and empty space is marked as 1 and 0 respectively in the grid.
/// For example,
/// ere is one obstacle in the middle of a 3 × 3 grid as illustrated below.
/// [
/// [0,0,0],
/// [0,1,0],
/// [0,0,0]
/// ]
/// e total number of unique paths is 2.
/// Note: m and n will be at most 100.
pub fn unique_paths_with_obstacles(grid: Vec<Vec<i32>>) -> i32 {
    if grid.first().is_none() || grid.first().unwrap().is_empty() {
        return 0;
    }
    let (n, first) = match grid.first() { Some(x) => (x.len(), *x.first().unwrap() ^ 1), None => return 0 };
    let mut f = vec![0; n];
    
    f[0] = first;
    grid.iter().fold(0, |_, y| {
        y.iter().zip(f.iter_mut()).enumerate().fold(0, |p, (i, (&ele, z))| {
            *z = if ele > 0 {0} else {*z + if i == 0 {0} else {p}};
            *z
        })
    })
}

struct Chessboard {
    col: Vec<bool>,
    p_diag: Vec<bool>,
    c_diag: Vec<bool>,
    queen_pos: Vec<usize>,
    res: Vec<Vec<String>>,
    count: i32,
    is_count: bool
}

trait ChessFinish {
    fn finish(&mut self);
}

impl Chessboard {
    fn new(n: usize, is_count: bool) -> Self {
        let (mut col, mut p_diag, mut c_diag, mut queen_pos) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        col.resize(n, false);
        p_diag.resize(n << 1, false);
        c_diag.resize(n << 1, false);
        queen_pos.resize(n, 0);
        
        Chessboard {
            col, p_diag, c_diag, queen_pos, res: Vec::new(), count: 0, is_count,
        }
    }
    
    fn result(self) -> (Option<i32>, Option<Vec<Vec<String>>>) {
        if self.is_count {(Some(self.count), None)} else {(None, Some(self.res))}
    }
    
    fn dfs(&mut self, row: usize) {
        if row == self.queen_pos.len() {
            if self.is_count {
                self.count += 1;
            } else {
                let mut tmp = Vec::with_capacity(self.queen_pos.len());
                self.queen_pos.iter().for_each(|&pos| {
                    let mut s = ".".repeat(pos);
                    s.push('Q');
                    s.push_str(".".repeat(self.queen_pos.len() - 1 - pos).as_str());
                    tmp.push(s);
                });
                self.res.push(tmp);
            }
            return;
        }

        for col in 00..self.queen_pos.len() {
            if !self.col[col] && !self.p_diag[row + col]
                && !self.c_diag[row + self.queen_pos.len() - col] {
                self.queen_pos[row] = col;
                self.col[col] = true;
                self.p_diag[row + col] = true;
                self.c_diag[row + self.queen_pos.len() - col] = true;

                self.dfs(row + 1);

                self.col[col] = false;
                self.p_diag[row + col] = false;
                self.c_diag[row + self.queen_pos.len() - col] = false;
            }
        }
    }
}

/// N-Queens
/// e n-queens puzzle is the problem of placing n queens on an n × n chessboard such that no two
/// queens aack each other.
/// Given an integer n, return all distinct solutions to the n-queens puzzle.
/// Each solution contains a distinct board configuration of the n-queens’ placement, where 'Q' and '.'
/// both indicate a queen and an empty space respectively.
/// For example, ere exist two distinct solutions to the 4-queens puzzle:
/// [
/// [".Q..", // Solution 1
/// "...Q",
/// "Q...",
/// "..Q."],
/// ["..Q.", // Solution 2
/// "Q...",
/// "...Q",
/// ".Q.."]
/// ]
pub fn solve_nqueens(n: i32) -> Vec<Vec<String>> {
    if n <= 0 {
        return vec![vec![]];
    }
    
    let mut chessboard = Chessboard::new(n as usize, false);
    
    chessboard.dfs(0);
 
    chessboard.result().1.unwrap()
}

/// N-Queens II
pub fn total_nqueens(n: i32) -> i32 {
    if n <= 0 {
        return 1;
    }

    let mut chessboard = Chessboard::new(n as usize, true);

    chessboard.dfs(0);

    chessboard.result().0.unwrap()
}

/// Restore IP Address
/// Given a string containing only digits, restore it by returning all possible valid IP address combinations.
/// For example: Given ”25525511135”,
/// return [”255.255.11.135”, ”255.255.111.35”]. (Order does not matter)
pub fn restore_ip_address(s: String) -> Vec<String> {
    struct F(Vec<String>, fn(&Vec<u8>, usize, usize, String, f: &mut F));
    let dfs = |s: &Vec<u8>, start: usize, step: usize, mut ip: String, f: &mut F| {
        if s.len() == start && step == 4 {
            ip.truncate(ip.len() - 1);
            (f.0).push(ip.clone());
            return;
        } else if (s.len() - start) < (4 - step) {
            return;
        } else if (s.len() - start) > (4 - step) * 3 {
            return;
        }
        
        let mut num = 0;
        for i in start..std::cmp::min(s.len(), start+3) {
            num = num * 10 + (s[i] - b'0') as i32;
            if num <= 255 {
                ip.push(s[i] as char);
                let mut tmp = ip.clone();
                tmp.push('.');
                (f.1)(s, i+1, step+1, tmp, f);
            }
            if num == 0 {break;}
        }
    };

    let s = s.chars().map(|x| { (x as u32) as u8}).collect();
    let mut f = F(Vec::new(), dfs);
    dfs(&s, 0, 0, String::new(), &mut f);
    
    f.0
}

/// Combination Sum
/// Given a set of candidate numbers (C) and a target number (T ), find all unique combinations in C where
/// the candidate numbers sums to T .
/// e same repeated number may be chosen from C unlimited number of times.
/// Note:
/// • All numbers (including target) will be positive integers.
/// • Elements in a combination (a1; a2; :::; ak) must be in non-descending order. (ie, a1 > a2 > ::: > ak).
/// • e solution set must not contain duplicate combinations.
/// For example, given candidate set 2,3,6,7 and target 7, A solution set is:
/// [7]
/// [2, 2, 3]
pub fn combination_sum(nums: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
    struct F(Vec<Vec<i32>>, fn(&Vec<i32>, i32, usize, &mut Vec<i32>, &mut F));
    let dfs = |nums: &Vec<i32>, gap: i32, start: usize, inter: &mut Vec<i32>, f: &mut F| {
        if gap == 0 {
            f.0.push(inter.clone());
            return;
        }
        
        for i in start..nums.len() {
            if gap < nums[i] {return;}
            inter.push(nums[i]);
            (f.1)(nums, gap-nums[i], i, inter, f);
            inter.pop();
        }
    };
    
    let mut nums = nums;
    nums.sort();
    let mut f = F(Vec::new(), dfs);
    let mut inter = Vec::new();
    dfs(&nums, target, 0, &mut inter, &mut f);
    
    f.0
}

/// Combination Sum II
/// Given a set of candidate numbers (C) and a target number (T ), find all unique combinations in C where
/// the candidate numbers sums to T .
/// e same repeated number may be chosen from C once number of times.
/// Note:
/// • All numbers (including target) will be positive integers.
/// • Elements in a combination (a1; a2; :::; ak) must be in non-descending order. (ie, a1 > a2 > ::: > ak).
/// • e solution set must not contain duplicate combinations.
/// For example, given candidate set 10,1,2,7,6,1,5 and target 8, A solution set is:
/// [1, 7]
/// [1, 2, 5]
/// [2, 6]
/// [1, 1, 6]
pub fn combination_sum_ii(nums: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
    struct F(Vec<Vec<i32>>, fn(&Vec<i32>, i32, usize, &mut Vec<i32>, &mut F));
    let dfs = |nums: &Vec<i32>, gap: i32, start: usize, inter: &mut Vec<i32>, f: &mut F| {
        if gap == 0 {
            f.0.push(inter.clone());
            return;
        }

        let mut pre: i32 = 0;
        for i in start..nums.len() {
            if pre == nums[i] {continue;}
            if gap < nums[i] {return;}
            
            pre = nums[i];
            inter.push(nums[i]);
            (f.1)(nums, gap-nums[i], i+1, inter, f);
            
            inter.pop();
        }
    };

    let mut nums = nums;
    nums.sort();
    let mut f = F(Vec::new(), dfs);
    let mut inter = Vec::new();
    dfs(&nums, target, 0, &mut inter, &mut f);

    f.0
}

/// Generate Parentheses
/// Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
/// For example, given n = 3, a solution set is:
/// "((()))", "(()())", "(())()", "()(())", "()()()"
pub fn generate_parentheses(n: i32) -> Vec<String> {
    struct F(Vec<String>, fn(i32, &mut String, i32, i32, &mut F));
    let generate = |n: i32, s: &mut String, l: i32, r: i32, f: &mut F| {
        if n == l {
            let mut tmp = s.clone();
            tmp.push_str(")".repeat((n-r) as usize).as_str());
            (f.0).push(tmp);
            return;
        }
        
        s.push('(');
        (f.1)(n, s, l+1, r, f);
        s.pop();
        
        if l > r {
            s.push(')');
            (f.1)(n, s, l, r+1, f);
            s.pop();
        }
    };
    
    if n <=0 {
        vec!["".to_string()]
    } else {
        let mut f = F(Vec::new(), generate);
        let mut inter = String::with_capacity((n << 1) as usize);
        generate(n, &mut inter, 0, 0, &mut f);
        
        f.0
    }
}

/// Sudoku Solver
/// Write a program to solve a Sudoku puzzle by filling the empty cells.
/// Empty cells are indicated by the character '.'.
/// You may assume that there will be only one unique solution.
pub fn solve_sudoku(board: &mut Vec<Vec<char>>) -> bool {
    let check = |b: &Vec<Vec<char>>, x: usize, y: usize| {
        for i in 0..9 { 
            if i != x && b[i][y] == b[x][y] {
                return false;
            }
        }
        
        for j in 0..9 {
            if j != y && b[x][j] == b[x][y] {
                return false;
            }
        }
        
        let (xs, xe, ys, ye) = (3*(x/3),3*(x/3 + 1), 3*(y/3), 3*(y/3+1));
        for i in xs..xe {
            for j in ys..ye {
                if i != x && j != y && b[i][j] == b[x][y] {
                    return false;
                }
            }
        }
        
        true
    };
    
    for i in 0..9 {
        for j in 0..9 {
            if board[i][j] == '.' {
                for k in 0..9 {
                    board[i][j] = (b'1' + k) as char;
                    if check(&*board, i, j) && solve_sudoku(board) {
                        return true;
                    }
                    board[i][j] = '.';
                }
                return false;
            }
        }
    }
    true
}

/// Word Search
/// Given a 2D board and a word, find if the word exists in the grid.
/// e word can be constructed from leers of sequentially adjacent cell, where "adjacent" cells are
/// those horizontally or vertically neighbouring. e same leer cell may not be used more than once.
/// For example, Given board =
/// [
/// ["ABCE"],
/// ["SFCS"],
/// ["ADEE"]
/// ]
/// word = "ABCCED", -> returns true,
/// word = "SEE", -> returns true,
/// word = "ABCB", -> returns false.
pub fn exist(board: Vec<Vec<char>>, word: String) -> bool {
    struct F(Vec<Vec<bool>>, fn(&Vec<Vec<char>>, &Vec<char>, usize, i32, i32, &mut F) -> bool);
    let dfs = |board: &Vec<Vec<char>>, word: &Vec<char>, idx: usize, x: i32, y: i32,
        f: &mut F| -> bool {
        if idx == word.len() {
            return true;
        }
        
        if x < 0 || y < 0 || x >= (board.len() as i32) || y >= (board.first().unwrap().len() as i32) {
            return false;
        }
        
        let (x, y) = (x as usize, y as usize);
        if (f.0)[x][y] {
            return false;
        }
        
        if board[x][y] != word[idx] {
            return false;
        }

        (f.0)[x][y] = true;
        let (x, y) = (x as i32, y as i32);
        let res = (f.1)(board, word, idx+1, x-1, y, f) ||
            (f.1)(board, word, idx+1, x+1,y, f) ||
            (f.1)(board, word, idx+1, x, y-1, f) ||
            (f.1)(board, word, idx+1, x, y+1, f);
        (f.0)[x as usize][y as usize] = false;
        res
    };
    
    if board.is_empty() {
        return word.is_empty();
    }
    
    let (m, n) = (board.len(), board.first().unwrap().len());
    let mut f = F(vec![vec![false; n]; m], dfs);
    let word: Vec<char> = word.chars().collect();
    let (m, n) = (m as i32, n as i32);
    for i in 0..m {
        for j in 0..n {
            if dfs(&board, &word, 0, i, j, &mut f) {
                return true;
            }
        }
    }
    
    false
}

#[cfg(test)]
mod tests {
    #[test]
    fn search_matrix() {
        let cases= [
            ((vec![vec![]], 1), false),
            ((vec![vec![1,2,3]],1), true),
        ];

        cases.iter().for_each(|x| {
            assert_eq!(x.1, super::search_matrix((x.0).0.to_vec(), (x.0).1), "cases: {:?}", x.0);
        });
    }
    
    #[test]
    fn ladder_length() {
        let cases = [
            (("ymain","oecij", vec!["ymann","yycrj","oecij","ymcnj","yzcrj","yycij","xecij","yecij","ymanj","yzcnj","ymain"]), 10),
            (("hit", "cog", vec!["hot", "dot", "dog", "lot", "log"]), 0),
        ];
        
        cases.iter().for_each(|x| {
            assert_eq!(x.1, super::ladder_length((x.0).0.to_string(), (x.0).1.to_string(),
                                                 (x.0).2.iter().map(|ele| {ele.to_string()}).collect()), 
                "cases: {:?}->{:?}", (x.0).0, (x.0).1);
        })
    }
    
    #[test]
    fn find_ladders() {
        let cases = [
            (("red", "tax",vec!["ted","tex","red","tax","tad","den","rex","pee"]), 4),
            (("hit", "cog", vec!["hot","dot","dog","lot","log","cog"]), 5),
            (("ymain","oecij", vec!["ymann","yycrj","oecij","ymcnj","yzcrj","yycij","xecij","yecij","ymanj","yzcnj","ymain"]), 10),
            (("hit", "cog", vec!["hot", "dot", "dog", "lot", "log"]), 0),
            (("sand","acne",
            vec!["slit","bunk","wars","ping","viva","wynn","wows","irks","gang","pool","mock","fort","heel","send","ship","cols","alec","foal","nabs","gaze","giza","mays","dogs","karo","cums","jedi","webb","lend","mire","jose","catt","grow","toss","magi","leis","bead","kara","hoof","than","ires","baas","vein","kari","riga","oars","gags","thug","yawn","wive","view","germ","flab","july","tuck","rory","bean","feed","rhee","jeez","gobs","lath","desk","yoko","cute","zeus","thus","dims","link","dirt","mara","disc","limy","lewd","maud","duly","elsa","hart","rays","rues","camp","lack","okra","tome","math","plug","monk","orly","friz","hogs","yoda","poop","tick","plod","cloy","pees","imps","lead","pope","mall","frey","been","plea","poll","male","teak","soho","glob","bell","mary","hail","scan","yips","like","mull","kory","odor","byte","kaye","word","honk","asks","slid","hopi","toke","gore","flew","tins","mown","oise","hall","vega","sing","fool","boat","bobs","lain","soft","hard","rots","sees","apex","chan","told","woos","unit","scow","gilt","beef","jars","tyre","imus","neon","soap","dabs","rein","ovid","hose","husk","loll","asia","cope","tail","hazy","clad","lash","sags","moll","eddy","fuel","lift","flog","land","sigh","saks","sail","hook","visa","tier","maws","roeg","gila","eyes","noah","hypo","tore","eggs","rove","chap","room","wait","lurk","race","host","dada","lola","gabs","sobs","joel","keck","axed","mead","gust","laid","ends","oort","nose","peer","kept","abet","iran","mick","dead","hags","tens","gown","sick","odis","miro","bill","fawn","sumo","kilt","huge","ores","oran","flag","tost","seth","sift","poet","reds","pips","cape","togo","wale","limn","toll","ploy","inns","snag","hoes","jerk","flux","fido","zane","arab","gamy","raze","lank","hurt","rail","hind","hoot","dogy","away","pest","hoed","pose","lose","pole","alva","dino","kind","clan","dips","soup","veto","edna","damp","gush","amen","wits","pubs","fuzz","cash","pine","trod","gunk","nude","lost","rite","cory","walt","mica","cart","avow","wind","book","leon","life","bang","draw","leek","skis","dram","ripe","mine","urea","tiff","over","gale","weir","defy","norm","tull","whiz","gill","ward","crag","when","mill","firs","sans","flue","reid","ekes","jain","mutt","hems","laps","piss","pall","rowe","prey","cull","knew","size","wets","hurl","wont","suva","girt","prys","prow","warn","naps","gong","thru","livy","boar","sade","amok","vice","slat","emir","jade","karl","loyd","cerf","bess","loss","rums","lats","bode","subs","muss","maim","kits","thin","york","punt","gays","alpo","aids","drag","eras","mats","pyre","clot","step","oath","lout","wary","carp","hums","tang","pout","whip","fled","omar","such","kano","jake","stan","loop","fuss","mini","byrd","exit","fizz","lire","emil","prop","noes","awed","gift","soli","sale","gage","orin","slur","limp","saar","arks","mast","gnat","port","into","geed","pave","awls","cent","cunt","full","dint","hank","mate","coin","tars","scud","veer","coax","bops","uris","loom","shod","crib","lids","drys","fish","edit","dick","erna","else","hahs","alga","moho","wire","fora","tums","ruth","bets","duns","mold","mush","swop","ruby","bolt","nave","kite","ahem","brad","tern","nips","whew","bait","ooze","gino","yuck","drum","shoe","lobe","dusk","cult","paws","anew","dado","nook","half","lams","rich","cato","java","kemp","vain","fees","sham","auks","gish","fire","elam","salt","sour","loth","whit","yogi","shes","scam","yous","lucy","inez","geld","whig","thee","kelp","loaf","harm","tomb","ever","airs","page","laud","stun","paid","goop","cobs","judy","grab","doha","crew","item","fogs","tong","blip","vest","bran","wend","bawl","feel","jets","mixt","tell","dire","devi","milo","deng","yews","weak","mark","doug","fare","rigs","poke","hies","sian","suez","quip","kens","lass","zips","elva","brat","cosy","teri","hull","spun","russ","pupa","weed","pulp","main","grim","hone","cord","barf","olav","gaps","rote","wilt","lars","roll","balm","jana","give","eire","faun","suck","kegs","nita","weer","tush","spry","loge","nays","heir","dope","roar","peep","nags","ates","bane","seas","sign","fred","they","lien","kiev","fops","said","lawn","lind","miff","mass","trig","sins","furl","ruin","sent","cray","maya","clog","puns","silk","axis","grog","jots","dyer","mope","rand","vend","keen","chou","dose","rain","eats","sped","maui","evan","time","todd","skit","lief","sops","outs","moot","faze","biro","gook","fill","oval","skew","veil","born","slob","hyde","twin","eloy","beat","ergs","sure","kobe","eggo","hens","jive","flax","mons","dunk","yest","begs","dial","lodz","burp","pile","much","dock","rene","sago","racy","have","yalu","glow","move","peps","hods","kins","salk","hand","cons","dare","myra","sega","type","mari","pelt","hula","gulf","jugs","flay","fest","spat","toms","zeno","taps","deny","swag","afro","baud","jabs","smut","egos","lara","toes","song","fray","luis","brut","olen","mere","ruff","slum","glad","buds","silt","rued","gelt","hive","teem","ides","sink","ands","wisp","omen","lyre","yuks","curb","loam","darn","liar","pugs","pane","carl","sang","scar","zeds","claw","berg","hits","mile","lite","khan","erik","slug","loon","dena","ruse","talk","tusk","gaol","tads","beds","sock","howe","gave","snob","ahab","part","meir","jell","stir","tels","spit","hash","omit","jinx","lyra","puck","laue","beep","eros","owed","cede","brew","slue","mitt","jest","lynx","wads","gena","dank","volt","gray","pony","veld","bask","fens","argo","work","taxi","afar","boon","lube","pass","lazy","mist","blot","mach","poky","rams","sits","rend","dome","pray","duck","hers","lure","keep","gory","chat","runt","jams","lays","posy","bats","hoff","rock","keri","raul","yves","lama","ramp","vote","jody","pock","gist","sass","iago","coos","rank","lowe","vows","koch","taco","jinn","juno","rape","band","aces","goal","huck","lila","tuft","swan","blab","leda","gems","hide","tack","porn","scum","frat","plum","duds","shad","arms","pare","chin","gain","knee","foot","line","dove","vera","jays","fund","reno","skid","boys","corn","gwyn","sash","weld","ruiz","dior","jess","leaf","pars","cote","zing","scat","nice","dart","only","owls","hike","trey","whys","ding","klan","ross","barb","ants","lean","dopy","hock","tour","grip","aldo","whim","prom","rear","dins","duff","dell","loch","lava","sung","yank","thar","curl","venn","blow","pomp","heat","trap","dali","nets","seen","gash","twig","dads","emmy","rhea","navy","haws","mite","bows","alas","ives","play","soon","doll","chum","ajar","foam","call","puke","kris","wily","came","ales","reef","raid","diet","prod","prut","loot","soar","coed","celt","seam","dray","lump","jags","nods","sole","kink","peso","howl","cost","tsar","uric","sore","woes","sewn","sake","cask","caps","burl","tame","bulk","neva","from","meet","webs","spar","fuck","buoy","wept","west","dual","pica","sold","seed","gads","riff","neck","deed","rudy","drop","vale","flit","romp","peak","jape","jews","fain","dens","hugo","elba","mink","town","clam","feud","fern","dung","newt","mime","deem","inti","gigs","sosa","lope","lard","cara","smug","lego","flex","doth","paar","moon","wren","tale","kant","eels","muck","toga","zens","lops","duet","coil","gall","teal","glib","muir","ails","boer","them","rake","conn","neat","frog","trip","coma","must","mono","lira","craw","sled","wear","toby","reel","hips","nate","pump","mont","died","moss","lair","jibe","oils","pied","hobs","cads","haze","muse","cogs","figs","cues","roes","whet","boru","cozy","amos","tans","news","hake","cots","boas","tutu","wavy","pipe","typo","albs","boom","dyke","wail","woke","ware","rita","fail","slab","owes","jane","rack","hell","lags","mend","mask","hume","wane","acne","team","holy","runs","exes","dole","trim","zola","trek","puma","wacs","veep","yaps","sums","lush","tubs","most","witt","bong","rule","hear","awry","sots","nils","bash","gasp","inch","pens","fies","juts","pate","vine","zulu","this","bare","veal","josh","reek","ours","cowl","club","farm","teat","coat","dish","fore","weft","exam","vlad","floe","beak","lane","ella","warp","goth","ming","pits","rent","tito","wish","amps","says","hawk","ways","punk","nark","cagy","east","paul","bose","solo","teed","text","hews","snip","lips","emit","orgy","icon","tuna","soul","kurd","clod","calk","aunt","bake","copy","acid","duse","kiln","spec","fans","bani","irma","pads","batu","logo","pack","oder","atop","funk","gide","bede","bibs","taut","guns","dana","puff","lyme","flat","lake","june","sets","gull","hops","earn","clip","fell","kama","seal","diaz","cite","chew","cuba","bury","yard","bank","byes","apia","cree","nosh","judo","walk","tape","taro","boot","cods","lade","cong","deft","slim","jeri","rile","park","aeon","fact","slow","goff","cane","earp","tart","does","acts","hope","cant","buts","shin","dude","ergo","mode","gene","lept","chen","beta","eden","pang","saab","fang","whir","cove","perk","fads","rugs","herb","putt","nous","vane","corm","stay","bids","vela","roof","isms","sics","gone","swum","wiry","cram","rink","pert","heap","sikh","dais","cell","peel","nuke","buss","rasp","none","slut","bent","dams","serb","dork","bays","kale","cora","wake","welt","rind","trot","sloe","pity","rout","eves","fats","furs","pogo","beth","hued","edam","iamb","glee","lute","keel","airy","easy","tire","rube","bogy","sine","chop","rood","elbe","mike","garb","jill","gaul","chit","dons","bars","ride","beck","toad","make","head","suds","pike","snot","swat","peed","same","gaza","lent","gait","gael","elks","hang","nerf","rosy","shut","glop","pain","dion","deaf","hero","doer","wost","wage","wash","pats","narc","ions","dice","quay","vied","eons","case","pour","urns","reva","rags","aden","bone","rang","aura","iraq","toot","rome","hals","megs","pond","john","yeps","pawl","warm","bird","tint","jowl","gibe","come","hold","pail","wipe","bike","rips","eery","kent","hims","inks","fink","mott","ices","macy","serf","keys","tarp","cops","sods","feet","tear","benz","buys","colo","boil","sews","enos","watt","pull","brag","cork","save","mint","feat","jamb","rubs","roxy","toys","nosy","yowl","tamp","lobs","foul","doom","sown","pigs","hemp","fame","boor","cube","tops","loco","lads","eyre","alta","aged","flop","pram","lesa","sawn","plow","aral","load","lied","pled","boob","bert","rows","zits","rick","hint","dido","fist","marc","wuss","node","smog","nora","shim","glut","bale","perl","what","tort","meek","brie","bind","cake","psst","dour","jove","tree","chip","stud","thou","mobs","sows","opts","diva","perm","wise","cuds","sols","alan","mild","pure","gail","wins","offs","nile","yelp","minn","tors","tran","homy","sadr","erse","nero","scab","finn","mich","turd","then","poem","noun","oxus","brow","door","saws","eben","wart","wand","rosa","left","lina","cabs","rapt","olin","suet","kalb","mans","dawn","riel","temp","chug","peal","drew","null","hath","many","took","fond","gate","sate","leak","zany","vans","mart","hess","home","long","dirk","bile","lace","moog","axes","zone","fork","duct","rico","rife","deep","tiny","hugh","bilk","waft","swig","pans","with","kern","busy","film","lulu","king","lord","veda","tray","legs","soot","ells","wasp","hunt","earl","ouch","diem","yell","pegs","blvd","polk","soda","zorn","liza","slop","week","kill","rusk","eric","sump","haul","rims","crop","blob","face","bins","read","care","pele","ritz","beau","golf","drip","dike","stab","jibs","hove","junk","hoax","tats","fief","quad","peat","ream","hats","root","flak","grit","clap","pugh","bosh","lock","mute","crow","iced","lisa","bela","fems","oxes","vies","gybe","huff","bull","cuss","sunk","pups","fobs","turf","sect","atom","debt","sane","writ","anon","mayo","aria","seer","thor","brim","gawk","jack","jazz","menu","yolk","surf","libs","lets","bans","toil","open","aced","poor","mess","wham","fran","gina","dote","love","mood","pale","reps","ines","shot","alar","twit","site","dill","yoga","sear","vamp","abel","lieu","cuff","orbs","rose","tank","gape","guam","adar","vole","your","dean","dear","hebe","crab","hump","mole","vase","rode","dash","sera","balk","lela","inca","gaea","bush","loud","pies","aide","blew","mien","side","kerr","ring","tess","prep","rant","lugs","hobo","joke","odds","yule","aida","true","pone","lode","nona","weep","coda","elmo","skim","wink","bras","pier","bung","pets","tabs","ryan","jock","body","sofa","joey","zion","mace","kick","vile","leno","bali","fart","that","redo","ills","jogs","pent","drub","slaw","tide","lena","seep","gyps","wave","amid","fear","ties","flan","wimp","kali","shun","crap","sage","rune","logs","cain","digs","abut","obit","paps","rids","fair","hack","huns","road","caws","curt","jute","fisk","fowl","duty","holt","miss","rude","vito","baal","ural","mann","mind","belt","clem","last","musk","roam","abed","days","bore","fuze","fall","pict","dump","dies","fiat","vent","pork","eyed","docs","rive","spas","rope","ariz","tout","game","jump","blur","anti","lisp","turn","sand","food","moos","hoop","saul","arch","fury","rise","diss","hubs","burs","grid","ilks","suns","flea","soil","lung","want","nola","fins","thud","kidd","juan","heps","nape","rash","burt","bump","tots","brit","mums","bole","shah","tees","skip","limb","umps","ache","arcs","raft","halo","luce","bahs","leta","conk","duos","siva","went","peek","sulk","reap","free","dubs","lang","toto","hasp","ball","rats","nair","myst","wang","snug","nash","laos","ante","opal","tina","pore","bite","haas","myth","yugo","foci","dent","bade","pear","mods","auto","shop","etch","lyly","curs","aron","slew","tyro","sack","wade","clio","gyro","butt","icky","char","itch","halt","gals","yang","tend","pact","bees","suit","puny","hows","nina","brno","oops","lick","sons","kilo","bust","nome","mona","dull","join","hour","papa","stag","bern","wove","lull","slip","laze","roil","alto","bath","buck","alma","anus","evil","dumb","oreo","rare","near","cure","isis","hill","kyle","pace","comb","nits","flip","clop","mort","thea","wall","kiel","judd","coop","dave","very","amie","blah","flub","talc","bold","fogy","idea","prof","horn","shoo","aped","pins","helm","wees","beer","womb","clue","alba","aloe","fine","bard","limo","shaw","pint","swim","dust","indy","hale","cats","troy","wens","luke","vern","deli","both","brig","daub","sara","sued","bier","noel","olga","dupe","look","pisa","knox","murk","dame","matt","gold","jame","toge","luck","peck","tass","calf","pill","wore","wadi","thur","parr","maul","tzar","ones","lees","dark","fake","bast","zoom","here","moro","wine","bums","cows","jean","palm","fume","plop","help","tuba","leap","cans","back","avid","lice","lust","polo","dory","stew","kate","rama","coke","bled","mugs","ajax","arts","drug","pena","cody","hole","sean","deck","guts","kong","bate","pitt","como","lyle","siam","rook","baby","jigs","bret","bark","lori","reba","sups","made","buzz","gnaw","alps","clay","post","viol","dina","card","lana","doff","yups","tons","live","kids","pair","yawl","name","oven","sirs","gyms","prig","down","leos","noon","nibs","cook","safe","cobb","raja","awes","sari","nerd","fold","lots","pete","deal","bias","zeal","girl","rage","cool","gout","whey","soak","thaw","bear","wing","nagy","well","oink","sven","kurt","etna","held","wood","high","feta","twee","ford","cave","knot","tory","ibis","yaks","vets","foxy","sank","cone","pius","tall","seem","wool","flap","gird","lore","coot","mewl","sere","real","puts","sell","nuts","foil","lilt","saga","heft","dyed","goat","spew","daze","frye","adds","glen","tojo","pixy","gobi","stop","tile","hiss","shed","hahn","baku","ahas","sill","swap","also","carr","manx","lime","debs","moat","eked","bola","pods","coon","lacy","tube","minx","buff","pres","clew","gaff","flee","burn","whom","cola","fret","purl","wick","wigs","donn","guys","toni","oxen","wite","vial","spam","huts","vats","lima","core","eula","thad","peon","erie","oats","boyd","cued","olaf","tams","secs","urey","wile","penn","bred","rill","vary","sues","mail","feds","aves","code","beam","reed","neil","hark","pols","gris","gods","mesa","test","coup","heed","dora","hied","tune","doze","pews","oaks","bloc","tips","maid","goof","four","woof","silo","bray","zest","kiss","yong","file","hilt","iris","tuns","lily","ears","pant","jury","taft","data","gild","pick","kook","colt","bohr","anal","asps","babe","bach","mash","biko","bowl","huey","jilt","goes","guff","bend","nike","tami","gosh","tike","gees","urge","path","bony","jude","lynn","lois","teas","dunn","elul","bonn","moms","bugs","slay","yeah","loan","hulk","lows","damn","nell","jung","avis","mane","waco","loin","knob","tyke","anna","hire","luau","tidy","nuns","pots","quid","exec","hans","hera","hush","shag","scot","moan","wald","ursa","lorn","hunk","loft","yore","alum","mows","slog","emma","spud","rice","worn","erma","need","bags","lark","kirk","pooh","dyes","area","dime","luvs","foch","refs","cast","alit","tugs","even","role","toed","caph","nigh","sony","bide","robs","folk","daft","past","blue","flaw","sana","fits","barr","riot","dots","lamp","cock","fibs","harp","tent","hate","mali","togs","gear","tues","bass","pros","numb","emus","hare","fate","wife","mean","pink","dune","ares","dine","oily","tony","czar","spay","push","glum","till","moth","glue","dive","scad","pops","woks","andy","leah","cusp","hair","alex","vibe","bulb","boll","firm","joys","tara","cole","levy","owen","chow","rump","jail","lapp","beet","slap","kith","more","maps","bond","hick","opus","rust","wist","shat","phil","snow","lott","lora","cary","mote","rift","oust","klee","goad","pith","heep","lupe","ivan","mimi","bald","fuse","cuts","lens","leer","eyry","know","razz","tare","pals","geek","greg","teen","clef","wags","weal","each","haft","nova","waif","rate","katy","yale","dale","leas","axum","quiz","pawn","fend","capt","laws","city","chad","coal","nail","zaps","sort","loci","less","spur","note","foes","fags","gulp","snap","bogs","wrap","dane","melt","ease","felt","shea","calm","star","swam","aery","year","plan","odin","curd","mira","mops","shit","davy","apes","inky","hues","lome","bits","vila","show","best","mice","gins","next","roan","ymir","mars","oman","wild","heal","plus","erin","rave","robe","fast","hutu","aver","jodi","alms","yams","zero","revs","wean","chic","self","jeep","jobs","waxy","duel","seek","spot","raps","pimp","adan","slam","tool","morn","futz","ewes","errs","knit","rung","kans","muff","huhs","tows","lest","meal","azov","gnus","agar","sips","sway","otis","tone","tate","epic","trio","tics","fade","lear","owns","robt","weds","five","lyon","terr","arno","mama","grey","disk","sept","sire","bart","saps","whoa","turk","stow","pyle","joni","zinc","negs","task","leif","ribs","malt","nine","bunt","grin","dona","nope","hams","some","molt","smit","sacs","joan","slav","lady","base","heck","list","take","herd","will","nubs","burg","hugs","peru","coif","zoos","nick","idol","levi","grub","roth","adam","elma","tags","tote","yaws","cali","mete","lula","cubs","prim","luna","jolt","span","pita","dodo","puss","deer","term","dolt","goon","gary","yarn","aims","just","rena","tine","cyst","meld","loki","wong","were","hung","maze","arid","cars","wolf","marx","faye","eave","raga","flow","neal","lone","anne","cage","tied","tilt","soto","opel","date","buns","dorm","kane","akin","ewer","drab","thai","jeer","grad","berm","rods","saki","grus","vast","late","lint","mule","risk","labs","snit","gala","find","spin","ired","slot","oafs","lies","mews","wino","milk","bout","onus","tram","jaws","peas","cleo","seat","gums","cold","vang","dewy","hood","rush","mack","yuan","odes","boos","jami","mare","plot","swab","borg","hays","form","mesh","mani","fife","good","gram","lion","myna","moor","skin","posh","burr","rime","done","ruts","pays","stem","ting","arty","slag","iron","ayes","stub","oral","gets","chid","yens","snub","ages","wide","bail","verb","lamb","bomb","army","yoke","gels","tits","bork","mils","nary","barn","hype","odom","avon","hewn","rios","cams","tact","boss","oleo","duke","eris","gwen","elms","deon","sims","quit","nest","font","dues","yeas","zeta","bevy","gent","torn","cups","worm","baum","axon","purr","vise","grew","govs","meat","chef","rest","lame"]),0),
        ];

        cases.iter().for_each(|x| {
            super::find_ladders((x.0).0.to_string(), (x.0).1.to_string(),
                                    (x.0).2.iter().map(|ele| {ele.to_string()}).collect());
        });
    }
    
    #[test]
    fn restore_ip_address() {
        let cases = [
            ("25525511135", vec!["255.255.111.35", "255.255.11.135"]),
        ];
        
        cases.iter().for_each(|x| {
            let mut left = x.1.clone();
            left.sort();
            let mut right = super::restore_ip_address(x.0.to_string());
            right.sort();
            assert_eq!(left, right, "case: {:?}", x.0);
        })
    }
    
    #[test]
    fn combination_sum() {
        let cases = [
            ((vec![8,7,4,3], 11), vec![vec![3,4,4],vec![3,8],vec![4,7]]),
        ];
        
        cases.iter().for_each(|x| {
            let mut left = x.1.clone();
            left.sort();
            let mut right = super::combination_sum((x.0).0.clone(), (x.0).1);
            right.sort();
            assert_eq!(left, right, "case: {:?}", x.0);
        })
    }
    
    #[test]
    fn exist() {
        let cases = [
            ((vec![vec!['A','B','C','E'],vec!['S','F','C','S'],vec!['A','D','E','E']], "SEE"), true),
            ((vec![vec!['a','b'],vec!['c','d']], "acdb"), true),
        ];
        
        cases.iter().for_each(|x| {
            assert_eq!(x.1, super::exist((x.0).0.clone(), (x.0).1.to_string()), "case: {:?}", x.0);
        })
    }
}