use leetcode_meta::{Difficulty, Tag, Topic};

#[test]
fn test_enum() {
    for (i, &name) in Topic::names().iter().enumerate() {
        let topic_i = Topic::try_from(i as u8).unwrap();
        let topic_str = Topic::try_from(name).unwrap();
        assert_eq!(topic_i, topic_str, "{} != {}", topic_i, topic_str);
    }

    for (i, &name) in Tag::names().iter().enumerate() {
        let tag_i = Tag::try_from(i as u8).unwrap();
        let tag_str = Tag::try_from(name).unwrap();
        assert_eq!(tag_i, tag_str, "{} != {}", tag_i, tag_str);
    }

    for (i, &name) in Difficulty::names().iter().enumerate() {
        let difficulty_i = Difficulty::try_from(i as u8).unwrap();
        let difficulty_str = Difficulty::try_from(name).unwrap();
        assert_eq!(
            difficulty_i, difficulty_str,
            "{} != {}",
            difficulty_i, difficulty_str
        );
    }
}
