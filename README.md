# Rust Word Embeddings

This repository provides a way to generate word embeddings for use in natural language processing using Rust. The program uses principal component analysis (PCA) to form reduced word embeddings from the co-occurrence matrix.
A [dataset of movie reviews](https://ai.stanford.edu/~amaas/data/sentiment/) is included in the repository for demonstration.

## Overview

The word embeddings are suitable for use in natural language processing, and are used in my [RustTransformer repository](https://github.com/goldstraw/WordEmbeddings). Sample word embeddings are included for convenience.

## Usage

To generate the word embeddings, you must have Rust and Cargo installed on your machine. After installing Rust and Cargo, you can clone this repository to your local machine.

To generate the word embeddings, use the following command:

```
$ cargo run --release
```

The program will prompt you for various settings such as dimensionality and the number of threads. Upon completion the word embeddings will be saved to an output JSON file.

## Further Reading

Original paper for dataset: [Learning Word Vectors for Sentiment Analysis](http://www.aclweb.org/anthology/P11-1015)

For more information about this project, read [my blog post on transformers](https://charliegoldstraw.com/articles/transformers/).

## License

The source code for the transformer is licensed under the GNU Affero General Public License v3.0 - see the `LICENSE` file for details.
