To use AI download train_dataset.csv, validation_dataset.csv, and PyTorch_Neural_AI.py

Then run the following script to train the AI

python PyTorch_Neural_AI.py --mode train \
  --train_csv train_dataset.csv \
  --val_csv validation_dataset.csv \
  --epochs 100

Randomly selects one game from the validation set and the AI generates a narrative on it and compares it with the target narrative.

python PyTorch_Neural_AI.py --mode generate \
  --input_csv validation_dataset.csv \
  --row random
