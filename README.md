# LSTM-SNLI

## Progress
1. Problem definition temporarily converted to Entailment vs No Entailment (contradiction and neutral)
	- Because CosineEmbeddingLoss() in pytorch only takes in true labels of -1 and 1
	- Different loss functions and additional settings in CosineEmbeddingLoss will be explored to include all three labels.
2. Due to large train times, small subsets (10,000 and 2,000) of the train and test sets were used.
3. Vocabulary is generated from the train subset using FastText Embeddings. The 'unknown' token maps to 0 index.
4. 'unknown' token used for testing but not for training. This can be easily achieved using a dropout layer at the start when training the model.
5. Basic LSTM model is used with very small embedding and hidden dimensions (6 and 4 respectively).
6. Best results after 20 epochs:
	- Train accuracy: 61%
	- Test accuracy: 52%
7. Higher accuracies may be achieved with larger embedding and hidden vector sizes.