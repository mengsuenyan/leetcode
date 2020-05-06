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

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

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
}