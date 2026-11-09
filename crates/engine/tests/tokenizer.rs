use std::path::PathBuf;

use gpt2_engine::Tokenizer;

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/gpt2")
}

fn tokenizer() -> Option<Tokenizer> {
    let dir = model_dir();
    if !dir.join("vocab.json").exists() || !dir.join("merges.txt").exists() {
        eprintln!("skipping tokenizer test: {} is missing", dir.display());
        return None;
    }
    Some(Tokenizer::load(&dir).expect("tokenizer loads"))
}

#[test]
fn encodes_known_ids() {
    let Some(tokenizer) = tokenizer() else {
        return;
    };
    assert_eq!(tokenizer.encode("Hello world"), [15496, 995]);
    assert_eq!(
        tokenizer.encode(" The quick brown fox"),
        [383, 2068, 7586, 21831]
    );
    assert_eq!(
        tokenizer.encode("Hello, my name is"),
        [15496, 11, 616, 1438, 318]
    );
    assert_eq!(
        tokenizer.encode("The quick brown fox jumps over the lazy dog."),
        [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13]
    );
    assert_eq!(
        tokenizer.encode("I'm testing, isn't it?"),
        [40, 1101, 4856, 11, 2125, 470, 340, 30]
    );
    assert_eq!(tokenizer.encode("don't"), [9099, 470]);
    assert_eq!(tokenizer.encode(" 's"), [705, 82]);
    assert_eq!(tokenizer.encode("Hello  world"), [15496, 220, 995]);
    assert_eq!(tokenizer.encode("  leading spaces"), [220, 3756, 9029]);
    assert_eq!(
        tokenizer.encode("123 4567 8.9"),
        [10163, 4153, 3134, 807, 13, 24]
    );
    assert_eq!(
        tokenizer.encode("café naïve 日本語"),
        [66, 1878, 2634, 41492, 10545, 245, 98, 17312, 105, 45739, 252]
    );
    assert_eq!(
        tokenizer.encode("\n\nNew paragraph\twith tab"),
        [198, 198, 3791, 7322, 197, 4480, 7400]
    );
    assert_eq!(tokenizer.encode("\n'ta"), [198, 470, 64]);
    assert_eq!(tokenizer.encode("\t's"), [197, 338]);
    assert_eq!(tokenizer.encode("\r\n're"), [201, 198, 821]);
    assert_eq!(tokenizer.encode("\u{a0}'ll"), [1849, 1183]);
    assert_eq!(tokenizer.encode("a'b'c"), [64, 6, 65, 6, 66]);
}

#[test]
fn decodes_to_the_original_text() {
    let Some(tokenizer) = tokenizer() else {
        return;
    };
    for text in [
        "Hello world",
        " The quick brown fox",
        "Hello, my name is",
        "I'm testing, isn't it?",
        "The quick brown fox jumps over the lazy dog.",
        "Hello  world",
        "  leading spaces",
        "café naïve 日本語",
        "\n\nNew paragraph\twith tab",
        "supercalifragilisticexpialidocious",
    ] {
        assert_eq!(tokenizer.decode(&tokenizer.encode(text)), text);
    }
    assert_eq!(tokenizer.decode(&[50256]), "<|endoftext|>");
}
