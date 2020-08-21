//! 搜索相关

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
}