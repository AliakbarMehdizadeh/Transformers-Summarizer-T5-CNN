import optuna
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
from evaluate import load

class TextSummarizationTrainer:
    def __init__(self, model_name="t5-small"):
        # Load dataset and metric
        self.dataset = load_dataset("cnn_dailymail", "3.0.0")
        # Access the training dataset
        train_dataset = self.dataset['train']

        # Example: Select a subset of the training dataset (first `subset_size` examples)
        self.dataset['train'] = train_dataset.select(range(10000))

        # Example: Filter the training dataset to keep only examples with articles shorter than `max_article_length`
        self.dataset['train'] = train_dataset.filter(lambda x: len(x['article'].split()) < 1000)
        self.dataset['validation'] = train_dataset.filter(lambda x: len(x['article'].split()) < 1000)
        self.dataset['test'] = train_dataset.filter(lambda x: len(x['article'].split()) < 1000)

        self.metric = load("bleu")

        # Initialize tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        # Tokenize dataset
        self.tokenized_datasets = self.dataset.map(self.preprocess_function, batched=True)

    def preprocess_function(self, examples):
        inputs = examples["article"]
        targets = examples["highlights"]
        model_inputs = self.tokenizer(inputs, max_length=512, padding="max_length", truncation=True)
        labels = self.tokenizer(targets, max_length=150, padding="max_length", truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(self, eval_preds):
        logits, labels = eval_preds
        predictions = self.tokenizer.batch_decode(logits, skip_special_tokens=True)
        references = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = self.metric.compute(predictions=predictions, references=references)
        return result

    def objective(self, trial):
        # Define hyperparameters to be tuned
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True)
        batch_size = trial.suggest_categorical('per_device_train_batch_size', [2,4])
        num_train_epochs = trial.suggest_int('num_train_epochs', 3, 10)

        training_args = TrainingArguments(
            output_dir="./results",    
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size, 
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            fp16=True  # Enable mixed precision to save memory
        )

        model = T5ForConditionalGeneration.from_pretrained(self.model_name)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["validation"],
            compute_metrics=self.compute_metrics,
        )

        eval_results = trainer.evaluate()
        return eval_results['eval_loss']  # Minimize the evaluation loss

    def tune_hyperparameters(self, n_trials=2):
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(direction='minimize', pruner=pruner)
        study.optimize(self.objective, n_trials=n_trials)
        self.best_params = study.best_params
        print("Best Hyperparameters: ", self.best_params)

    def train_best_model(self):
        if not hasattr(self, 'best_params'):
            raise RuntimeError("Hyperparameters have not been tuned yet. Please run `tune_hyperparameters` first.")

        # Retrieve the best hyperparameters
        training_args = TrainingArguments(
            output_dir="./results",    
            per_device_train_batch_size=self.best_params['per_device_train_batch_size'],
            per_device_eval_batch_size=self.best_params['per_device_train_batch_size'],
            num_train_epochs=self.best_params['num_train_epochs'],
            learning_rate=self.best_params['learning_rate'],
            weight_decay=0.01,
            evaluation_strategy="epoch"
        )

        # Load the best model
        model = T5ForConditionalGeneration.from_pretrained(self.model_name)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["validation"],
            compute_metrics=self.compute_metrics,
        )

        # Train the model with the best hyperparameters
        trainer.train()

        # Save the model
        model.save_pretrained("best_t5_model")
        self.tokenizer.save_pretrained("best_t5_model")
