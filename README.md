We will add to this for Chartier


python PyTorch_Neural_AI.py --mode train \
  --train_csv train_dataset.csv \
  --val_csv validation_dataset.csv \
  --epochs 100

python PyTorch_Neural_AI.py --mode generate \
  --input_csv validation_dataset.csv \
  --row random
