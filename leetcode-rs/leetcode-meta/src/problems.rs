use crate::{Difficulty, Id, Problem, Tag, Tags, Topic};
use regex::Regex;
use std::collections::HashSet;
use std::ops::BitOr;

pub struct Iter<'a, T: AsRef<Problem>> {
    iter: std::slice::Iter<'a, T>,
}

impl<'a, T: AsRef<Problem>> Iterator for Iter<'a, T> {
    type Item = &'a Problem;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|p| p.as_ref())
    }
}

impl<'a, T: AsRef<Problem>> Iter<'a, T> {
    fn new(iter: std::slice::Iter<'a, T>) -> Self {
        Self { iter }
    }
}

pub struct IterMut<'a, T: AsRef<Problem> + AsMut<Problem>> {
    iter: std::slice::IterMut<'a, T>,
}

impl<'a, T: AsRef<Problem> + AsMut<Problem>> Iterator for IterMut<'a, T> {
    type Item = &'a mut Problem;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|p| p.as_mut())
    }
}

impl<'a, T: AsRef<Problem> + AsMut<Problem>> IterMut<'a, T> {
    fn new(iter: std::slice::IterMut<'a, T>) -> Self {
        Self { iter }
    }
}

pub struct Problems<T: AsRef<Problem>> {
    problems: Vec<T>,
}

impl<T: AsRef<Problem>> Problems<T> {
    pub const fn new() -> Self {
        Self { problems: vec![] }
    }

    pub fn all(&self) -> Problems<&Problem> {
        Problems::<&Problem> {
            problems: self.iter().collect(),
        }
    }

    fn from_vec(problems: Vec<T>) -> Self {
        Self { problems }
    }

    pub fn iter(&self) -> Iter<'_, T> {
        Iter::new(self.problems.iter())
    }

    pub fn exist(&self, id: Id) -> bool {
        self.iter().any(|p| p.id() == id)
    }

    /// insert the `problem` to the `self`, but not remove repeat problems.
    pub fn push(&mut self, problem: T) {
        self.problems.push(problem);
    }

    pub fn len(&self) -> usize {
        self.problems.len()
    }

    pub fn is_empty(&self) -> bool {
        self.problems.is_empty()
    }

    pub fn sort_by_id(&mut self) {
        self.problems.sort_by_key(|x| x.as_ref().id())
    }

    /// This will remove repeated problem by it's id.
    pub fn purge(&mut self) {
        let mut tmp = HashSet::with_capacity(self.len());
        for p in self.problems.iter() {
            tmp.insert(p.as_ref().id());
        }

        let purged_len = tmp.len();

        let mut mark = 0;
        let ids = self.iter().map(|p| p.id()).collect::<Vec<_>>();
        for (i, id) in ids.iter().enumerate() {
            if tmp.contains(id) {
                self.problems.swap(mark, i);
                tmp.remove(id);
                mark += 1;
            }
        }

        self.problems.truncate(purged_len);
    }

    pub fn find_by_id(&self, id: Id) -> Problems<&Problem> {
        Problems::from_vec(self.iter().filter(|p| p.id() == id).collect())
    }

    pub fn find_by_difficulty(&self, difficulty: Difficulty) -> Problems<&Problem> {
        Problems::from_vec(
            self.iter()
                .filter(|p| p.difficulty() == difficulty)
                .collect(),
        )
    }

    pub fn find_by_topic(&self, topic: Topic) -> Problems<&Problem> {
        Problems::from_vec(self.iter().filter(|p| p.topic() == topic).collect())
    }

    pub fn find_by_tag(&self, tag: Tag) -> Problems<&Problem> {
        Problems::from_vec(
            self.iter()
                .filter(|p| p.tags().iter().any(|&x| x == tag))
                .collect(),
        )
    }

    pub fn find_by_tags(&self, mut tags: Tags) -> Problems<&Problem> {
        tags.sort_by_key(|&x| x as u8);
        tags.dedup();
        Problems::from_vec(
            self.iter()
                .filter(|p| tags.iter().all(|x| p.tags().contains(x)))
                .collect(),
        )
    }

    pub fn find_by_title(&self, re: &Regex) -> Problems<&Problem> {
        Problems::from_vec(self.iter().filter(|p| re.is_match(p.title())).collect())
    }

    pub fn find_by_solution(&self, re: &Regex) -> Problems<&Problem> {
        Problems::from_vec(
            self.iter()
                .filter(|p| !p.solution().is_empty() && re.is_match(p.solution()))
                .collect(),
        )
    }

    pub fn find_by_note(&self, re: Regex) -> Problems<&Problem> {
        Problems::from_vec(
            self.iter()
                .filter(|p| !p.note().is_empty() && re.is_match(p.note()))
                .collect(),
        )
    }
}

impl<T: AsRef<Problem> + AsMut<Problem>> Problems<T> {
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut::new(self.problems.iter_mut())
    }

    pub fn find_by_id_mut(&mut self, id: Id) -> Problems<&mut Problem> {
        Problems::from_vec(self.iter_mut().filter(|p| p.id() == id).collect())
    }

    pub fn find_by_difficulty_mut(&mut self, difficulty: Difficulty) -> Problems<&mut Problem> {
        Problems::from_vec(
            self.iter_mut()
                .filter(|p| p.difficulty() == difficulty)
                .collect(),
        )
    }

    pub fn find_by_topic_mut(&mut self, topic: Topic) -> Problems<&mut Problem> {
        Problems::from_vec(self.iter_mut().filter(|p| p.topic() == topic).collect())
    }

    pub fn find_by_tag_mut(&mut self, tag: Tag) -> Problems<&mut Problem> {
        Problems::from_vec(
            self.iter_mut()
                .filter(|p| p.tags().iter().any(|&x| x == tag))
                .collect(),
        )
    }

    pub fn find_by_title_mut(&mut self, re: &Regex) -> Problems<&mut Problem> {
        Problems::from_vec(self.iter_mut().filter(|p| re.is_match(p.title())).collect())
    }

    pub fn find_by_solution_mut(&mut self, re: &Regex) -> Problems<&mut Problem> {
        Problems::from_vec(
            self.iter_mut()
                .filter(|p| !p.solution().is_empty() && re.is_match(p.solution()))
                .collect(),
        )
    }

    pub fn find_by_note_mut(&mut self, re: Regex) -> Problems<&mut Problem> {
        Problems::from_vec(
            self.iter_mut()
                .filter(|p| !p.note().is_empty() && re.is_match(p.note()))
                .collect(),
        )
    }
}

impl Problems<Problem> {
    pub fn insert(&mut self, problem: Problem) {
        self.problems.push(problem);
    }
}

impl<T: AsRef<Problem>> Default for Problems<T> {
    fn default() -> Self {
        Self {
            problems: Vec::new(),
        }
    }
}

impl<T: AsRef<Problem>> BitOr for Problems<T> {
    type Output = Problems<T>;

    fn bitor(mut self, mut rhs: Self) -> Self::Output {
        self.problems.append(&mut rhs.problems);
        self
    }
}

impl<T: AsRef<Problem> + Clone> BitOr<&Problems<T>> for Problems<T> {
    type Output = Problems<T>;

    fn bitor(mut self, rhs: &Problems<T>) -> Self::Output {
        for p in rhs.problems.iter() {
            self.problems.push(p.clone());
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::{Difficulty, Id, Problem, Problems, Tag, Topic};

    fn insert() -> Problems<Problem> {
        let mut p = Problems::new();

        let r = (0..100u32)
            .map(|i| {
                if i < 50 {
                    i
                } else {
                    rand::random::<u32>().saturating_add(50)
                }
            })
            .collect::<Vec<_>>();

        for i in r {
            let mut t = Problem::new(Id::from(i));
            t.set_title(format!("{}-{}", i, i.wrapping_mul(i)))
                .set_topic(match i % 4 {
                    0 => Topic::Algorithm,
                    1 => Topic::Concurrency,
                    2 => Topic::Database,
                    3 => Topic::Shell,
                    _ => unreachable!(),
                })
                .set_difficulty(match i % 3 {
                    0 => Difficulty::Easy,
                    1 => Difficulty::Medium,
                    2 => Difficulty::Hard,
                    _ => unreachable!(),
                })
                .push_tag(Tag::HashTable)
                .push_tag(Tag::Array);
            p.insert(t);
        }

        p
    }

    #[test]
    fn test_insert_and_purge() {
        let mut ps = insert();
        let l1 = ps.len();
        ps.insert(Problem::new(2));
        ps.insert(Problem::new(9));
        ps.insert(Problem::new(39));
        assert_eq!(l1 + 3, ps.len(), "insert failed");

        ps.purge();
        assert_eq!(l1, ps.len(), "purge failed");
    }
}
