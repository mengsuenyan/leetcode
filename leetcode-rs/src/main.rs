use clap::{Arg, Command};
use leetcode_rs::prelude::*;
use regex::Regex;
use std::collections::HashMap;
use std::io::Write;

fn main() {
    let pretty_val_name = |s: String| {
        s.replace(',', " |")
            .replace('[', "")
            .replace(']', " ...")
            .replace('"', "")
    };

    let matches = Command::new("leetcode")
        .arg(
            Arg::new("id")
                .long("id")
                .value_name("Integer")
                .required(false)
                .takes_value(true)
                .multiple_values(false)
                .help("search by id"),
        )
        .arg(
            Arg::new("title")
                .long("title")
                .value_name("Regex")
                .required(false)
                .takes_value(true)
                .multiple_values(false)
                .help("search by title"),
        )
        .arg(
            Arg::new("difficulty")
                .long("difficulty")
                .value_name(
                    pretty_val_name(format!(
                        "{:?}",
                        Difficulty::names().into_iter().take(3).collect::<Vec<_>>()
                    ))
                    .as_str(),
                )
                .required(false)
                .takes_value(true)
                .multiple_values(false)
                .help("search by difficulty"),
        )
        .arg(
            Arg::new("topic")
                .long("topic")
                .value_name(
                    pretty_val_name(format!(
                        "{:?}",
                        Topic::names().into_iter().take(3).collect::<Vec<_>>()
                    ))
                    .as_str(),
                )
                .required(false)
                .takes_value(true)
                .multiple_values(false)
                .help("search by topic"),
        )
        .arg(
            Arg::new("tag")
                .long("tag")
                .value_name(
                    pretty_val_name(format!(
                        "{:?}",
                        Tag::names().into_iter().take(3).collect::<Vec<_>>()
                    ))
                    .as_str(),
                )
                .required(false)
                .takes_value(true)
                .multiple_values(true)
                .help("search by tag"),
        )
        .arg(
            Arg::new("solution")
                .long("solution")
                .value_name("Regex")
                .required(false)
                .takes_value(true)
                .multiple_values(false)
                .help("search by solution"),
        )
        .arg(
            Arg::new("note")
                .long("note")
                .value_name("Regex")
                .required(false)
                .takes_value(true)
                .multiple_values(false)
                .help("search by note"),
        )
        .get_matches();

    let (id, title, difficulty, topic, tag, solution, note) = (
        matches
            .is_present("id")
            .then(|| matches.value_of_t::<Id>("id").unwrap()),
        matches
            .is_present("title")
            .then(|| matches.value_of_t::<Regex>("title").unwrap()),
        matches
            .is_present("difficulty")
            .then(|| matches.value_of_t::<Difficulty>("difficulty").unwrap()),
        matches
            .is_present("topic")
            .then(|| matches.value_of_t::<Topic>("topic").unwrap()),
        matches
            .is_present("tag")
            .then(|| matches.values_of_t::<Tag>("tag").unwrap()),
        matches
            .is_present("solution")
            .then(|| matches.value_of_t::<Regex>("solution").unwrap()),
        matches
            .is_present("note")
            .then(|| matches.value_of_t::<Regex>("note").unwrap()),
    );

    match PROBLEMS.read() {
        Ok(p) => {
            let (id_s, p) = if let Some(id) = id {
                (format!("Id: {id}\n"), p.find_by_id(id))
            } else {
                (Default::default(), p.all())
            };
            let (diff_s, p) = if let Some(d) = difficulty {
                (format!("Difficulty: {d}\n"), p.find_by_difficulty(d))
            } else {
                (Default::default(), p)
            };
            let (topic_s, p) = if let Some(t) = topic {
                (format!("Topic: {t}\n"), p.find_by_topic(t))
            } else {
                (Default::default(), p)
            };
            let (tags_s, p) = if let Some(t) = tag {
                (format!("Tags: {:?}\n", t), p.find_by_tags(t))
            } else {
                (Default::default(), p)
            };
            let (title_s, p) = if let Some(t) = title {
                (format!("Title: {t}\n"), p.find_by_title(&t))
            } else {
                (Default::default(), p)
            };
            let (solution_s, p) = if let Some(s) = solution {
                (format!("Solution: {s}\n"), p.find_by_solution(&s))
            } else {
                (Default::default(), p)
            };
            let (note_s, p) = if let Some(n) = note {
                (format!("Note: {n}\n"), p.find_by_note(&n))
            } else {
                (Default::default(), p)
            };

            let n = p.len();
            println!("Total searched {n} items by the condition of:\n{id_s}{title_s}{diff_s}{topic_s}{tags_s}{solution_s}{note_s}");
            let mut input = String::new();

            loop {
                input.clear();
                print!(">: ");
                std::io::stdout().flush().unwrap();
                match std::io::stdin().read_line(&mut input) {
                    Ok(_) => {}
                    Err(e) => {
                        eprintln!("{}", e);
                        continue;
                    }
                }

                let content = input
                    .trim()
                    .split('=')
                    .map(|s| s.trim())
                    .collect::<Vec<_>>();
                if content.len() != 1 && content.len() != 2 {
                    eprintln!("Invalid input: {:?}", content);
                    continue;
                }

                let mut cmds = HashMap::with_capacity(cmd::CMD.len());
                cmd::CMD.iter().for_each(|s| {
                    cmds.insert(s.0, s.1);
                });

                match cmds.get(content[0]) {
                    None => {
                        eprintln!(
                            "Only support cmd: {:?}",
                            cmd::CMD.iter().map(|x| x.0).collect::<Vec<_>>()
                        );
                    }
                    Some(call) => {
                        call(&p, if content.len() == 2 { content[1] } else { "" });
                    }
                }
            }
        }
        Err(e) => {
            panic!("{}", e);
        }
    }
}

mod cmd {
    use leetcode_meta::{Difficulty, Id, Problem, Problems, Tag, Topic};
    use std::str::FromStr;

    macro_rules! enter_key {
        ($Input: ident) => {
            $Input.clear();
            match std::io::stdin().read_line(&mut $Input) {
                Ok(_) => {
                    if $Input.trim() == "exit" {
                        break;
                    }
                }
                Err(e) => {
                    eprintln!("{}", e);
                    break;
                }
            }
        };
    }

    #[allow(clippy::type_complexity)]
    pub const CMD: [(&str, fn(&Problems<&Problem>, &str), &str); 8] = [
        ("all", all, "all=[items every page], eg: all, all=10"),
        ("id", id, "id=[ids split by space], eg: id, id=1 2 3"),
        (
            "title",
            title,
            "title=[items every page], eg: title, title=10",
        ),
        (
            "topic",
            topic,
            "topic=[Topic;items every page], eg: topic, topic=Algorithm;10",
        ),
        (
            "tag",
            tag,
            "tag=[Tag;items every page], eg: tag, tag=string;10",
        ),
        (
            "difficulty",
            difficulty,
            "difficulty=[Difficulty;items every page], eg: difficulty, difficulty=easy;10",
        ),
        ("exit", exit, "exit program"),
        ("help", help, "help"),
    ];

    fn help(_problems: &Problems<&Problem>, para: &str) {
        if para.is_empty() {
            for e in CMD.iter() {
                println!("{:16} {}", e.0, e.2);
            }
        } else {
            for e in CMD.iter() {
                if para == e.0 {
                    println!("{:16}, {}", e.0, e.2);
                }
            }
        }
    }

    fn all(problems: &Problems<&Problem>, para: &str) {
        if para.is_empty() {
            problems.iter().for_each(|p| {
                println!("{}\n", p);
            });
        } else {
            match usize::from_str(para.trim()) {
                Ok(page) => {
                    let (mut input, mut n) = (String::new(), 0usize);
                    while n < problems.len() {
                        for p in problems.iter().skip(n).take(page) {
                            println!("{}\n", p);
                        }
                        n += page;
                        enter_key!(input);
                    }
                }
                Err(e) => {
                    println!("Items invalid {}", e);
                }
            }
        }
    }

    fn id(problems: &Problems<&Problem>, para: &str) {
        if para.is_empty() {
            print!("[");
            problems
                .iter()
                .map(|x| x.id())
                .enumerate()
                .for_each(|(i, x)| {
                    if i != problems.len().saturating_sub(1) {
                        print!("{}, ", x);
                    } else {
                        print!("{}", x)
                    }
                });
            println!("]");
        } else {
            para.split(' ')
                .map(|s| {
                    let s = s.trim();
                    if s.is_empty() {
                        Id::MAX
                    } else {
                        match Id::from_str(s) {
                            Ok(i) => i,
                            Err(e) => {
                                eprintln!("{}", e);
                                Id::MAX
                            }
                        }
                    }
                })
                .for_each(|s| {
                    for p in problems.find_by_id(s).iter() {
                        println!("{}\n", p);
                    }
                })
        };
    }

    fn title(problems: &Problems<&Problem>, para: &str) {
        if para.is_empty() {
            print!("[");
            problems.iter().for_each(|p| {
                print!("{}.{}; ", p.id(), p.title());
            });
            println!("]");
        } else {
            match usize::from_str(para.trim()) {
                Ok(page) => {
                    let (mut input, mut n) = (String::new(), 0usize);
                    while n < problems.len() {
                        for p in problems.iter().skip(n).take(page) {
                            println!("{}.{}", p.id(), p.title());
                        }
                        n += page;
                        enter_key!(input);
                    }
                }
                Err(e) => {
                    println!("Items invalid {}", e);
                }
            }
        }
    }

    macro_rules! fn_enum {
        ($Name: ident, $En: ty, $Find: ident) => {
            fn $Name(problems: &Problems<&Problem>, para: &str) {
                if para.is_empty() {
                    for name in <$En>::names() {
                        println!(
                            "{} `{}` problems",
                            problems.$Find(<$En>::from_str(name).unwrap()).len(),
                            name
                        );
                    }
                } else {
                    let p = para.split(';').map(|s| s.trim()).collect::<Vec<_>>();
                    let problems = if !p.is_empty() {
                        match <$En>::from_str(p[0]) {
                            Ok(d) => problems.$Find(d),
                            Err(e) => {
                                eprintln!("{}", e);
                                return;
                            }
                        }
                    } else {
                        eprintln!("Invalid para {}", para);
                        return;
                    };

                    let page = if p.len() > 1 {
                        match usize::from_str(p[1]) {
                            Ok(n) => n,
                            Err(e) => {
                                eprintln!("{}", e);
                                return;
                            }
                        }
                    } else {
                        problems.len()
                    };

                    let (mut input, mut n) = (String::new(), 0usize);
                    while n < problems.len() {
                        for p in problems.iter().skip(n).take(page) {
                            println!("{}.{}", p.id(), p.title());
                        }
                        n += page;
                        enter_key!(input);
                    }
                }
            }
        };
    }

    fn_enum!(topic, Topic, find_by_topic);
    fn_enum!(difficulty, Difficulty, find_by_difficulty);
    fn_enum!(tag, Tag, find_by_tag);

    fn exit(_problems: &Problems<&Problem>, _para: &str) {
        std::process::exit(0);
    }
}
