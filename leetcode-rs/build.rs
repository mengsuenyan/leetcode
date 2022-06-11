use rustversion::since;

#[since(1.61.0)]
#[derive(Debug)]
pub struct NeedToInstallRustVerGreatThan1_61_0;

fn main() {
    println!("{:?}", NeedToInstallRustVerGreatThan1_61_0 {});
}
