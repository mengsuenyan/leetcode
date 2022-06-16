# LeetCode

## LeetCode-rs

```shell
cd leetcode-rs
cargo build --release
```

```text
./target/release/leetcode-rs --help
leetcode 

USAGE:
    leetcode-rs [OPTIONS]

OPTIONS:
        --difficulty <Easy | Medium | Hard ...>       search by difficulty
    -h, --help                                        Print help information
        --id <Integer>                                search by id
        --note <Regex>                                search by note
        --solution <Regex>                            search by solution
        --tag <Array | String | HashTable ...>...     search by tag
        --title <Regex>                               search by title
        --topic <Algorithm | Database | Shell ...>    search by topic
```

```text
./target/release/leetcode-rs
>: help
all              all=[items every page], eg: all, all=10
id               id=[ids split by space], eg: id, id=1 2 3
title            title=[items every page], eg: title, title=10
topic            topic=[Topic;items every page], eg: topic, topic=Algorithm;10
tag              tag=[Tag;items every page], eg: tag, tag=string;10
difficulty       difficulty=[Difficulty;items every page], eg: difficulty, difficulty=easy;10
exit             exit program
help             help
```