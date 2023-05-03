use std::collections::HashMap;

struct Review {
    review: String,
    sentiment: bool,
}

/// Clean the review by removing all non-alphanumeric characters
fn clean_review(review: String) -> String {
    // "I love this movie! It's so good."
    // => "i love this movie it's so good "
    let mut clean_review: String = String::new();
    let mut in_word = false;
    let mut current_word = String::new();
    review.replace("<br", "");
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
    println!("{}", clean_review);
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

fn build_co_occurrence_matrix() {

}

fn main() {
    let imdb_dataset = load_imdb_dataset("imdb_dataset.csv");
    let vocab = build_vocab(&imdb_dataset);
    let co_occurrence_window = 3;
    let co_occurrence_matrix = HashMap::new();
    for review in imdb_dataset {
        let mut words = review.review.split_whitespace();
        let mut index = 0;
        for word in words {
            let mut index2 = 0;
            for word2 in words {
                let dist = (index - index2).abs();
                if dist <= co_occurrence_window {
                    let count = co_occurrence_matrix.entry(word).or_insert(HashMap::new());
                    let weight = 1.0 - (dist as f32 / co_occurrence_window as f32);
                    let count2 = count.entry(word2).or_insert(0.0);
                    *count2 += weight;
                }
                index2 += 1;
            }
            index += 1;
        }
    }
}
