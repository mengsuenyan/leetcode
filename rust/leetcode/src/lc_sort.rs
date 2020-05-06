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
}
