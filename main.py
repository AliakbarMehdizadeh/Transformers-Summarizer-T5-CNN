from TextSummarizationTrainer import TextSummarizationTrainer


if __name__ == "__main__":

    trainer = TextSummarizationTrainer()
    trainer.tune_hyperparameters(n_trials=10)
    trainer.train_best_model()
