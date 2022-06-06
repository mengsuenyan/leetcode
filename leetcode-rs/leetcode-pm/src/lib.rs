use leetcode_meta::{Difficulty, Id, Problem, Tag, Tags, Topic};
use proc_macro2::{Span, TokenStream, TokenTree};
use quote::ToTokens;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use syn::{Item, Result, Type};

#[derive(Hash)]
struct ParseItemType(Item);

impl ParseItemType {
    fn new(tokens: proc_macro::TokenStream) -> Result<Self> {
        syn::parse::<Item>(tokens).map(Self)
    }

    fn ident_name(&self) -> String {
        match &self.0 {
            Item::Fn(x) => x.sig.ident.to_string(),
            Item::Struct(x) => x.ident.to_string(),
            Item::Impl(x) => match x.self_ty.as_ref() {
                Type::Path(y) => y.path.segments.last().unwrap().ident.to_string(),
                y => {
                    panic!("Current doesn't support the type for ItemImpl: {:?}", y)
                }
            },
            x => {
                panic!("Current doesn't support the item: {:?}", x)
            }
        }
    }

    fn get_init_name(&self) -> String {
        let mut hasher = DefaultHasher::new();
        "init_lpm_".hash(&mut hasher);
        self.hash(&mut hasher);

        format!("{}_{:0>16x}", self.ident_name(), hasher.finish())
    }
}

impl ToTokens for ParseItemType {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.0.to_tokens(tokens)
    }
}

/// Only used for function
///
/// Example:
/// ```rust
///
/// #[inject_description(
/// problems="val",
/// id="1",
/// title="test",
/// difficulty="Easy",
/// topic="test",
/// tags="test1, test2",
/// note="ff",
/// )]
/// fn foo() {
/// }
/// ```
#[proc_macro_attribute]
pub fn inject_description(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let (problems, mut problem) = parse_attr(attr);

    let solution = prettyplease::unparse(&syn::parse(item.clone()).unwrap());
    problem.set_solution(solution);

    let ast = ParseItemType::new(item).unwrap();
    if problem.title().is_empty() {
        problem.set_title(ast.ident_name());
    }

    if let Some(problems) = problems {
        let name = syn::Ident::new(ast.get_init_name().as_str(), Span::call_site());

        let problems = syn::Ident::new(problems.as_str(), Span::call_site());
        let init_insert = quote::quote! {
            #[ctor]
            fn #name() {
                match #problems.write() {
                    Ok(mut ps) => {
                        let mut p = #problem;
                        p.set_file(file!().to_string()).set_line(line!());
                        ps.insert(p);
                    }
                    Err(e) => {
                        panic!("{:?}", e);
                    }
                }
            }
        };

        let mut ts = ast.into_token_stream();
        ts.extend(init_insert);

        ts.into()
    } else {
        ast.into_token_stream().into()
    }
}

// attr item must be with the syntax of `Ident=LitStr`, and split by the `,`;
//
// Example:
// ```txt
//     id="1",
//     topic="Algorithm",
// ```
fn parse_attr(attr: proc_macro::TokenStream) -> (Option<String>, Problem) {
    let attr = TokenStream::from(attr);

    const PUNCT_EQ: char = '=';
    const PUNCT_COMMA: char = ',';

    let mut g: Vec<Vec<TokenTree>> = vec![vec![]];
    for token_tree in attr {
        match token_tree {
            group @ TokenTree::Group(_) => {
                g.last_mut().unwrap().push(group);
            }
            ident @ TokenTree::Ident(_) => {
                g.last_mut().unwrap().push(ident);
            }
            TokenTree::Punct(punct) => match punct.as_char() {
                PUNCT_COMMA => g.push(vec![]),
                PUNCT_EQ => {
                    g.last_mut().unwrap().push(TokenTree::Punct(punct));
                }
                others => {
                    panic!("Invalid punct: {}", others);
                }
            },
            literal @ TokenTree::Literal(_) => {
                g.last_mut().unwrap().push(literal);
            }
        }
    }

    // the last `,` is allowed.
    if let Some(last) = g.last() {
        if last.is_empty() {
            g.pop();
        }
    }

    for group in g.iter() {
        if group.len() != 3 {
            panic!("Invalid format: the attr item must with the syntax of `Ident=LiteralString` and split by the `,`");
        }
    }

    let problems = parse_string(&g, "problems");
    let id = parse_id(&g).unwrap_or_else(|| panic!("The attr of id is required"));
    let title =
        parse_string(&g, "title").unwrap_or_else(|| panic!("The attr of title is required"));
    let note = parse_string(&g, "note").unwrap_or_default();
    let difficulty = parse_difficulty(&g);
    let topic = parse_topic(&g);
    let tags = parse_tags(&g);

    let mut problem = Problem::new(id);
    problem
        .set_title(title)
        .set_difficulty(difficulty)
        .set_topic(topic)
        .set_note(note)
        .append_tags(tags);

    (problems, problem)
}

// id="value"
fn parse_id(g: &[Vec<TokenTree>]) -> Option<Id> {
    for group in g.iter().rev() {
        match (&group[0], &group[1], &group[2]) {
            (TokenTree::Ident(ident), TokenTree::Punct(punct), TokenTree::Literal(literal)) => {
                if ident.to_string().to_lowercase().eq("id") {
                    let val = literal
                        .to_string()
                        .trim_start_matches('"')
                        .trim_end_matches('"')
                        .trim()
                        .parse::<u32>();
                    if punct.as_char() != '=' || val.is_err() {
                        panic!(
                            "Invalid Id attr: {}{}{} {}",
                            ident,
                            punct,
                            literal,
                            val.err().map(|e| format!(", {}", e)).unwrap_or_default()
                        );
                    } else {
                        return val.ok().map(Id::from);
                    }
                }
            }
            (ident, punct, literal) => {
                panic!("Invalid format: {}{}{}", ident, punct, literal);
            }
        }
    }

    None
}

// title="value"
// or
// note="value"
fn parse_string(g: &[Vec<TokenTree>], key: &str) -> Option<String> {
    for group in g.iter().rev() {
        match (&group[0], &group[1], &group[2]) {
            (TokenTree::Ident(ident), TokenTree::Punct(punct), TokenTree::Literal(literal)) => {
                if ident.to_string().to_lowercase().eq(key) {
                    if punct.as_char() != '=' {
                        panic!("Invalid {} attr: {}{}{}", key, ident, punct, literal);
                    } else {
                        return Some(
                            literal
                                .to_string()
                                .trim_start_matches('"')
                                .trim_end_matches('"')
                                .to_string(),
                        );
                    }
                }
            }
            (ident, punct, literal) => {
                panic!("Invalid format: {}{}{}", ident, punct, literal);
            }
        }
    }

    None
}

// difficulty="value"
fn parse_difficulty(g: &[Vec<TokenTree>]) -> Difficulty {
    for group in g.iter().rev() {
        match (&group[0], &group[1], &group[2]) {
            (TokenTree::Ident(ident), TokenTree::Punct(punct), TokenTree::Literal(literal)) => {
                if ident.to_string().to_lowercase().eq("difficulty") {
                    if punct.as_char() != '=' {
                        panic!("Invalid difficulty attr: {}{}{}", ident, punct, literal);
                    } else {
                        let val = literal.to_string();
                        match Difficulty::try_from(
                            val.trim_start_matches('"').trim_end_matches('"').trim(),
                        ) {
                            Ok(d) => {
                                return d;
                            }
                            Err(e) => {
                                panic!("Invalid difficulty due to: {}", e);
                            }
                        }
                    }
                }
            }
            (ident, punct, literal) => {
                panic!("Invalid format: {}{}{}", ident, punct, literal);
            }
        }
    }

    Difficulty::Unknown
}

// topic="value"
fn parse_topic(g: &[Vec<TokenTree>]) -> Topic {
    for group in g.iter().rev() {
        match (&group[0], &group[1], &group[2]) {
            (TokenTree::Ident(ident), TokenTree::Punct(punct), TokenTree::Literal(literal)) => {
                if ident.to_string().to_lowercase().eq("topic") {
                    if punct.as_char() != '=' {
                        panic!("Invalid topic attr: {}{}{}", ident, punct, literal);
                    } else {
                        let val = literal.to_string();
                        match Topic::try_from(
                            val.trim_start_matches('"').trim_end_matches('"').trim(),
                        ) {
                            Ok(d) => {
                                return d;
                            }
                            Err(e) => {
                                panic!("Invalid topic due to: {}", e);
                            }
                        }
                    }
                }
            }
            (ident, punct, literal) => {
                panic!("Invalid format: {}{}{}", ident, punct, literal);
            }
        }
    }

    Topic::Unknown
}

// tag="value"
// tags="value1, value2"
fn parse_tags(g: &[Vec<TokenTree>]) -> Tags {
    let mut buf = HashSet::new();

    for group in g.iter() {
        match (&group[0], &group[1], &group[2]) {
            (TokenTree::Ident(ident), TokenTree::Punct(punct), TokenTree::Literal(literal)) => {
                if punct.as_char() != '=' {
                    panic!("Invalid tag/tags attr: {}{}{}", ident, punct, literal);
                }

                if ident.to_string().to_lowercase().eq("tag") {
                    let s = literal.to_string();
                    if let Ok(tag) =
                        Tag::try_from(s.trim_start_matches('"').trim_end_matches('"').trim())
                    {
                        buf.insert(tag);
                    }
                } else if ident.to_string().to_lowercase().eq("tags") {
                    let s = literal.to_string();
                    s.trim_start_matches('"')
                        .trim_end_matches('"')
                        .split(',')
                        .for_each(|e| {
                            if let Ok(tag) = Tag::try_from(e.trim()) {
                                buf.insert(tag);
                            }
                        });
                } else {
                    continue;
                };
            }
            (ident, punct, literal) => {
                panic!("Invalid format: {}{}{}", ident, punct, literal);
            }
        }
    }

    buf.into_iter().collect::<Vec<_>>()
}
