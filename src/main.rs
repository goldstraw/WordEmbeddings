use std::collections::HashMap;

struct Review {
    review: String,
    sentiment: bool,
}

/// Clean the review by removing all non-alphanumeric characters
fn clean_review(mut review: String) -> String {
    // "I love this movie! It's so good."
    // => "i love this movie it's so good "
    let mut clean_review: String = String::new();
    let mut in_word = false;
    let mut current_word = String::new();
    review = review.replace("<br", "");
    for character in review.chars() {
        if character.is_alphanumeric() || (character == '\'' && in_word) {
            if !in_word {
                in_word = true;
            }
            current_word.push(character);
        } else {
            if in_word {
                in_word = false;
                if current_word != "" {
                    current_word = current_word.to_lowercase();
                    current_word.push(' ');
                    clean_review.push_str(&current_word);
                    current_word = String::new();
                }
            }
        }
    }
    if current_word != "" {
        current_word = current_word.to_lowercase();
        current_word.push(' ');
        clean_review.push_str(&current_word);
    }
    clean_review.to_string()
}

fn load_imdb_dataset(path: &str) -> Vec<Review> {
    let mut imdb_dataset = Vec::new();
    let mut reader = csv::Reader::from_path(path).unwrap();
    for result in reader.records() {
        let record = result.unwrap();
        let imdb_review = Review {
            review: clean_review(record[0].to_string()),
            sentiment: if record[1].to_string() == "positive" {true} else {false},
        };
        imdb_dataset.push(imdb_review);
        if imdb_dataset.len() == 1000 {
            break;
        }
    }
    imdb_dataset
}

fn build_vocab(dataset: &Vec<Review>) -> Vec<String> {
    let mut vocab: Vec<String> = Vec::new();
    for review in dataset {
        for word in review.review.split_whitespace() {
            if !vocab.contains(&word.to_string()) {
                vocab.push(word.to_string());
            }
        }
    }
    vocab
}

fn build_co_occurrence_matrix(vocab: &Vec<String>, imdb_dataset: &Vec<Review>) -> Vec<Vec<f32>> {
    let word_to_index: HashMap<String, usize> = vocab.iter().enumerate().map(|(i, x)| (x.to_string(), i)).collect();
    println!("{}", word_to_index["movie"]); // 356
    println!("{}", word_to_index["good"]); // 464
    let co_occurrence_window = 3;
    let mut co_occurrence_matrix = vec![vec![0.0; vocab.len()]; vocab.len()];
    for review in imdb_dataset {
        let words = review.review.split_whitespace();
        let mut index = 0;
        for word in words.clone() {
            let mut index2 = 0;
            for word2 in words.clone() {
                let dist = (index as i32 - index2 as i32).abs();
                if dist <= co_occurrence_window {
                    let weight = 1.0 - (dist as f32 / co_occurrence_window as f32);
                    co_occurrence_matrix[word_to_index[word]][word_to_index[word2]] += weight;
                }
                index2 += 1;
            }
            index += 1;
        }
    }
    co_occurrence_matrix
}

fn main() {
    let imdb_dataset = load_imdb_dataset("imdb_dataset.csv");
    let vocab = build_vocab(&imdb_dataset);
    let co_occurrence_matrix = build_co_occurrence_matrix(&vocab, &imdb_dataset);
    let movie = co_occurrence_matrix[356].clone();
    let good = co_occurrence_matrix[464].clone();

    for (i, word) in vocab.iter().enumerate() {
        if movie[i] > 0.0 {
            println!("movie: {} {}", word, movie[i]);
        }
        if good[i] > 0.0 {
            println!("good: {} {}", word, good[i]);
        }
    }
}
