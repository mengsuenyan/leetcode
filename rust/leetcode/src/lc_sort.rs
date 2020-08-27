//! 排序相关

/// Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.  
/// Note:  
/// The number of elements initialized in nums1 and nums2 are m and n respectively.  
/// You may assume that nums1 has enough space (size that is greater or equal to m + n) to hold additional elements from nums2.  
/// 来源：力扣（LeetCode）  
/// 链接：https://leetcode-cn.com/problems/merge-sorted-array  
/// 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。  
/// O( nlog(n) )
pub fn merge(nums1: &mut Vec<i32>, m: i32, nums2: &mut Vec<i32>, n: i32) {
    nums1.truncate(m as usize);
    nums2.truncate(n as usize);
    nums1.append(nums2);
    nums1.sort();
}

/// First Missing Positive
/// 
/// Given an unsorted integer array, find the first missing positive integer.
/// For example, Given [1,2,0] return 3, and [3,4,-1,1] return 2.
/// Your algorithm should run in O(n) time and uses constant space.
pub fn first_missing_positive(mut nums: Vec<i32>) -> i32 {
    // tc: O(n), sc: O(1)
    let sort = |v: &mut Vec<i32>| {
        (0..v.len()).for_each(|i| {
            while v[i] != (i + 1) as i32 {
                let j = if v[i] <= 0 || v[i] > (v.len() as i32) || v[i] == v[(v[i] as usize) - 1] {
                    break;
                } else {
                    v[i] as usize - 1
                };
                v.swap(i, j);
            }
        });
    };
    
    sort(&mut nums);
    for i in 0..(nums.len() as i32) {
        if nums[i as usize] != i + 1 {
            return i + 1;
        }
    }
    
    nums.len() as i32 + 1
}

/// Sort Colors
/// 
/// Given an array with n objects colored red, white or blue, sort them so that objects of the same color
/// are adjacent, with the colors in the order red, white and blue.
/// Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.
/// Note: You are not suppose to use the library’s sort function for this problem.
/// Follow up:
/// A rather straight forward solution is a two-pass algorithm using counting sort.
/// First, iterate the array counting number of 0’s, 1’s, and 2’s, then overwrite array with total number of
/// 0’s, then 1’s and followed by 2’s.
/// Could you come up with an one-pass algorithm using only constant space?
pub fn sort_colors(a: &mut Vec<i32>) {
    let (mut r, mut b) = (0, a.len() - 1);
    let mut i = 0;
    
    while i < (b + 1) {
        if a[i] == 0 {
            a.swap(i, r);
            i += 1;
            r += 1;
        } else if a[i] == 2 {
            a.swap(i, b);
            b -= 1;
        } else {
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn merge() {
        let mut cases = [
            (vec![1,2,3,0,0,0], 3,
                vec![2,5,6], 3,
                vec![1,2,2,3,5,6]
            ),
        ];
        
        for c in cases.iter_mut() {
            super::merge(&mut c.0, c.1, &mut c.2, c.3);
            assert_eq!(c.0, c.4);
        }
    }
    
    #[test]
    fn first_missing_positive() {

        let cases = [
            (vec![1,2,0],3),
            (vec![3,4,-1,1],2),
            (vec![7,8,9,11,12],1),
        ];

        cases.iter().for_each(|x| {
            assert_eq!(x.1, super::first_missing_positive(x.0.to_vec()), "cases: {:?}", x.0);
        });
    }
}
