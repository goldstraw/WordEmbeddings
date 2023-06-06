// https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c
// https://builtin.com/data-science/step-step-explanation-principal-component-analysis

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::fs::File;
use std::io::prelude::*;
use rand::Rng;
use serde::{Serialize, Deserialize};

struct Review {
    review: String,
    _sentiment: bool,
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
            _sentiment: if record[1].to_string() == "positive" {true} else {false},
        };
        imdb_dataset.push(imdb_review);
        if imdb_dataset.len() == 5000 {
            break;
        }
    }
    imdb_dataset
}

fn build_vocab(dataset: &Vec<Review>) -> Vec<String> {
    let mut vocab: Vec<String> = Vec::new();
    let mut word_counts: HashMap<String, usize> = HashMap::new();
    for review in dataset {
        for word in review.review.split_whitespace() {
            if !word_counts.contains_key(&word.to_string()) {
                word_counts.insert(word.to_string(), 0);
            } else {
                word_counts.insert(word.to_string(), word_counts[&word.to_string()]+1);
            }
            if word_counts[&word.to_string()] == 50 {
                vocab.push(word.to_string());
            }
        }
    }

    vocab
}

fn build_co_occurrence_matrix(vocab: &Vec<String>, imdb_dataset: &Vec<Review>) -> Vec<Vec<f32>> {
    let word_to_index: HashMap<String, usize> = vocab.iter().enumerate().map(|(i, x)| (x.to_string(), i)).collect();
    let co_occurrence_window = 4;
    let mut co_occurrence_matrix = vec![vec![0.0; vocab.len()]; vocab.len()];
    for review in imdb_dataset {
        let words = review.review.split_whitespace();
        let mut filtered_words: Vec<&str> = Vec::new();
        for word in words {
            if vocab.contains(&word.to_string()) {
                filtered_words.push(word);
            }
        }

        let mut index = 0;
        for word in filtered_words.clone() {
            let mut index2 = 0;
            for word2 in filtered_words.clone() {
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

fn power_iteration(cov: &mut Vec<Vec<f32>>, num_iterations: usize, num_eigenvectors: usize) -> Vec<Vec<f32>>{
    // Uses the power iteration algorithm to compute N eigenvectors.

    let mut eigenvectors: Vec<Vec<f32>> = Vec::new();

    for _ in 0..num_eigenvectors {
        // Generate random vector
        let mut eigenvector: Vec<f32> = vec![0.0; cov.len()];
        let mut rng = rand::thread_rng();
        for i in 0..cov.len() {
            eigenvector[i] = rng.gen();
        }

        for _ in 0..num_iterations {
            // Calculate dot product of covariance matrix and eigenvector
            let mut new_eigenvector: Vec<f32> = vec![0.0; cov.len()];
            for j in 0..cov.len() {
                for k in 0..cov.len() {
                    new_eigenvector[j] += cov[j][k] * eigenvector[k]
                }
            }

            // Calculate norm of result
            let mut sq_sum = 0.0;
            for j in 0..cov.len() {
                sq_sum += f32::powi(new_eigenvector[j], 2);
            }
            let norm = f32::powf(sq_sum, 0.5);

            // Normalise result
            for j in 0..cov.len() {
                eigenvector[j] = new_eigenvector[j] / norm;
            }
        }
        // Calculate eigenvalue from eigenvector
        let mut temp = vec![0.0; cov.len()];
        for i in 0..cov.len() {
            for j in 0..cov.len() {
                temp[i] += cov[i][j] * eigenvector[j]
            }
        }

        let mut eigenvalue = 0.0;
        for i in 0..cov.len() {
            eigenvalue += eigenvector[i] * temp[i];
        }

        eigenvectors.push(eigenvector.clone());

        // Redirect matrix to find next eigenvector
        for i in 0..cov.len() {
            for j in 0..cov.len() {
                cov[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
            }
        }
    }

    eigenvectors
}

fn pca(mut matrix: Vec<Vec<f32>>, num_components: usize) -> Vec<Vec<f32>> {
    println!("{}", matrix.len());
    // Apply principle component analysis (PCA) on a matrix.

    // Normalise data
    for i in 0..matrix.len() {
        let mut total = 0.0;
        for j in 0..matrix.len() {
            total += matrix[i][j];
        }
        let mean = total / matrix.len() as f32;
        let mut dists_from_mean = 0.0;
        for j in 0..matrix[i].len() {
            dists_from_mean += f32::powi(matrix[i][j] - mean, 2);
        }
        let stdev = f32::powf(dists_from_mean / matrix[i].len() as f32, 0.5);

        for j in 0..matrix.len() {
            matrix[i][j] = (matrix[i][j] - mean) / stdev;
        }
    }

    // Find covariance matrix
    let num_threads = 16;
    let matrix_len = matrix.len();
    let matrix = Arc::new(matrix.clone());
    let covariance_matrix = Arc::new(Mutex::new(vec![vec![0.0; matrix_len]; matrix_len]));

    let mut handles = vec![];
    for i in 0..num_threads {
        let matrix = Arc::clone(&matrix);
        let covariance_matrix = Arc::clone(&covariance_matrix);
        let col_per_thread = (matrix_len as f32 / num_threads as f32).ceil() as usize;
        let handle = thread::spawn(move || {
            for j in 0..col_per_thread {
                let index = i * col_per_thread + j;
                if index < matrix_len {
                    let mut inner_vector = vec![0.0; matrix_len];
                    for k in 0..matrix_len {
                        let mut sum = 0.0;
                        for l in 0..matrix_len {
                            sum += matrix[index][l] * matrix[k][l];
                        }
                        let covariance = sum / matrix_len as f32;
                        inner_vector[k] = covariance;
                    }

                    let mut cov_lock = covariance_matrix.lock().unwrap();
                    for k in 0..matrix_len {
                        cov_lock[index][k] = inner_vector[k];
                    }
                }
            }
        });
        handles.push(handle);
    }

    // Wait for all threads to finish
    for handle in handles {
        handle.join().unwrap();
    }

    // Compute eigenvectors and eigenvalues
    let mut cov_lock = covariance_matrix.lock().unwrap();
    let eigenvectors = power_iteration(&mut cov_lock, 10, num_components);

    let mut reduced: Vec<Vec<f32>> = Vec::new();
    for i in 0..matrix.len() {
        let mut embedding: Vec<f32> = Vec::new();
        for j in 0..num_components {
            let mut component: f32 = 0.0;
            for k in 0..matrix.len() {
                component += matrix[i][k] * eigenvectors[j][k]
            }
            embedding.push(component);
        }
        reduced.push(embedding);
    }

    reduced
}

#[derive(Serialize, Deserialize)]
struct WordEmbeddings {
    data: HashMap<String, Vec<f32>>
}

fn main() {
    let imdb_dataset = load_imdb_dataset("imdb_dataset.csv");
    let vocab = build_vocab(&imdb_dataset);
    let co_occurrence_matrix = build_co_occurrence_matrix(&vocab, &imdb_dataset);
    let reduced = pca(co_occurrence_matrix, 200);
    let mut word_embeddings = WordEmbeddings {data: HashMap::new()};
    for i in 0..vocab.len() {
        word_embeddings.data.insert(vocab[i].clone(), reduced[i].clone());
    }
    let serialized = serde_json::to_string(&word_embeddings).unwrap();
    let mut file = File::create("data.json").expect("Unable to create file");
    file.write_all(serialized.as_bytes()).expect("Unable to write");
}
