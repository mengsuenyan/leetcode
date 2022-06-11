use proc_macro2::{Ident, Punct, Spacing, Span, TokenStream};
use quote::TokenStreamExt;
use std::fmt::{Debug, Display, Error, Formatter, Write};
use std::str::FromStr;

macro_rules! impl_enum {
    ($Name: ident, $($Ele: ident),+) => {
        #[derive(Eq, PartialEq, Debug, Copy, Clone, Hash)]
        #[repr(u8)]
        pub enum $Name {
            $($Ele),+,
            Unknown
        }

        impl $Name {
            pub const NUMS: usize = $Name::Unknown as usize + 1;

            pub const fn names() -> [&'static str; Self::NUMS] {
                [$(stringify!($Ele)),+, "Unknown"]
            }
        }

        impl TryFrom<u8> for $Name {
            type Error = String;

            fn try_from(n: u8) -> Result<Self, Self::Error> {
                match n {
                    $(a if a == $Name::$Ele as u8 => Ok($Name::$Ele)),+,
                    a if a == Self::Unknown as u8 => Ok(Self::Unknown),
                    _ => {
                        Err(format!("{} is not recognition, so convert to {}", n, Self::Unknown))
                    }
                }
            }
        }

        impl Display for $Name {
            fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
                write!(f, "{:?}", self)
            }
        }

        impl Default for $Name {
            fn default() -> $Name {
                <$Name>::Unknown
            }
        }

        impl TryFrom<&str> for $Name {
            type Error = String;

            fn try_from(val: &str) -> Result<Self, Self::Error> {
                let names = Self::names();
                let sl = val.to_lowercase();

                for (i, name) in names.iter().enumerate() {
                    if sl == name.to_lowercase() {
                        return Self::try_from(i as u8);
                    }
                }

                Err(format!("{} is not recognition, so convert to {}", val, Self::Unknown))
            }
        }

        impl FromStr for $Name {
            type Err = String;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                Self::try_from(s)
            }
        }

        impl quote::ToTokens for $Name {
            fn to_tokens(&self, tokens: &mut TokenStream) {
                match self {
                    $(Self::$Ele => {
                        tokens.append(Ident::new(stringify!($Name), Span::call_site()));
                        tokens.append(Punct::new(':', Spacing::Joint));
                        tokens.append(Punct::new(':', Spacing::Alone));
                        tokens.append(Ident::new(stringify!($Ele), Span::call_site()));
                    }),+,
                    Self::Unknown => {
                        tokens.append(Ident::new(stringify!($Name), Span::call_site()));
                        tokens.append(Punct::new(':', Spacing::Joint));
                        tokens.append(Punct::new(':', Spacing::Alone));
                        tokens.append(Ident::new("Unknown", Span::call_site()));
                    }
                }
            }
        }
    }
}

impl_enum!(Topic, Algorithm, Database, Shell, Concurrency);
impl_enum!(Difficulty, Easy, Medium, Hard);
impl_enum!(
    Tag,
    Array,
    String,
    HashTable,
    DynamicProgramming,
    Math,
    Sorting,
    DepthFirstSearch,
    Greedy,
    DataBase,
    BreadthFirstSearch,
    Tree,
    BinarySearch,
    Matrix,
    BinaryTree,
    TwoPointers,
    BitManipulation,
    Stack,
    Design,
    Heap,
    Graph,
    Simulation,
    BackTracking,
    PrefixSum,
    Counting,
    SlidingWindow,
    LinkedList,
    UnionFind,
    OrderedSet,
    Recursion,
    BinarySearchTree,
    Trie,
    MonotonicStack,
    DivideAndConquer,
    BitMask,
    Enumeration,
    Queue,
    Geometry,
    Memoization,
    TopologicalSort,
    SegmentTree,
    GameTheory,
    HashFunction,
    BinaryIndexedTree,
    Interactive,
    RollingHash,
    DataStream,
    StringMatching,
    ShortestPath,
    Combinatorics,
    Randomized,
    NumberTheory,
    MonotonicQueue,
    Iterator,
    MergeSort,
    Concurrency,
    Brainteaser,
    ProbabilityAndStatistics,
    DoublyLinkedList,
    QuickSelect,
    BucketSort,
    SuffixArray,
    MinimumSpanningTree,
    CountingSort,
    Shell,
    LineSeep,
    ReservoirSampling,
    EulerianCircuit,
    StronglyConnectedComponent,
    RadixSort,
    RejectionSampling,
    BiconnectedComponent,
    Backtracking
);

pub type Id = u32;
pub type Tags = Vec<Tag>;

#[derive(Clone)]
pub struct Problem {
    id: Id,
    title: String,
    difficulty: Difficulty,
    topic: Topic,
    tags: Tags,
    solution: String,
    note: String,
    file: String,
    line: u32,
}

impl Display for Problem {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}. {}", self.id, self.title)?;
        writeln!(f, "Difficulty: {}, Topic: {}", self.difficulty, self.topic)?;

        f.write_str("Tags: [")?;
        let mut is_first = true;
        for tag in self.tags.iter() {
            if is_first {
                write!(f, "{}", tag)?;
                is_first = false;
            } else {
                write!(f, ", {}", tag)?;
            }
        }
        f.write_str("]\n")?;

        writeln!(f, "Note: {}", self.note)?;
        writeln!(f, "File: {}, Line: {}", self.file, self.line)?;
        f.write_str("Solution as follows:\n")?;
        write!(f, "{}", self.solution)?;
        f.write_char('\n')
    }
}

impl Debug for Problem {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl AsRef<Problem> for Problem {
    fn as_ref(&self) -> &Problem {
        self
    }
}

impl AsMut<Problem> for Problem {
    fn as_mut(&mut self) -> &mut Problem {
        self
    }
}

impl PartialEq for Problem {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl quote::ToTokens for Problem {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let Problem {
            id,
            title,
            difficulty,
            topic,
            tags,
            solution,
            note,
            file,
            line,
        } = self.clone();
        tokens.extend(quote::quote! {{
            let mut p = Problem::new(Id::from(#id));
            p.set_title(#title.to_string())
                .set_difficulty(#difficulty)
                .set_topic(#topic)
                .append_tags(vec![#(#tags,)*])
                .set_solution(#solution.to_string())
                .set_note(#note.to_string())
                .set_file(#file.to_string())
                .set_line(#line);
            p
        }});
    }
}

impl Problem {
    pub fn new(id: Id) -> Self {
        Self {
            id,
            title: String::default(),
            difficulty: Difficulty::default(),
            topic: Topic::default(),
            tags: Default::default(),
            solution: Default::default(),
            note: Default::default(),
            file: Default::default(),
            line: Default::default(),
        }
    }

    pub fn id(&self) -> Id {
        self.id
    }

    pub fn title(&self) -> &str {
        self.title.as_str()
    }

    pub fn set_title(&mut self, title: String) -> &mut Self {
        self.title = title;
        self
    }

    pub fn difficulty(&self) -> Difficulty {
        self.difficulty
    }

    pub fn set_difficulty(&mut self, difficulty: Difficulty) -> &mut Self {
        self.difficulty = difficulty;
        self
    }

    pub fn topic(&self) -> Topic {
        self.topic
    }

    pub fn set_topic(&mut self, topic: Topic) -> &mut Self {
        self.topic = topic;
        self
    }

    pub fn tags(&self) -> &[Tag] {
        self.tags.as_slice()
    }

    pub fn push_tag(&mut self, tag: Tag) -> &mut Self {
        if !self.tags.iter().any(|&x| x == tag) {
            self.tags.push(tag);
        }

        self
    }

    pub fn append_tags(&mut self, tags: Tags) -> &mut Self {
        for tag in tags {
            if !self.tags.iter().any(|&x| x == tag) {
                self.tags.push(tag);
            }
        }
        self
    }

    pub fn solution(&self) -> &str {
        self.solution.as_str()
    }

    pub fn set_solution(&mut self, solution: String) -> &mut Self {
        self.solution = solution;
        self
    }

    pub fn note(&self) -> &str {
        self.note.as_str()
    }

    pub fn set_note(&mut self, note: String) -> &mut Self {
        self.note = note;
        self
    }

    pub fn set_file(&mut self, file: String) -> &mut Self {
        self.file = file;
        self
    }

    pub fn set_line(&mut self, line: u32) -> &mut Self {
        self.line = line;
        self
    }
}
