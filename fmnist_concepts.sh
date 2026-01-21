python3 learn_concepts_dataset.py --dataset-name="cfMNIST" --backbone-name="resnet18_fmnist" --C 0.001 0.01 0.1 1.0 10.0 --n-samples=100 --out-dir=concept_bank/fMNIST --device cpu
python3 learn_concepts_dataset.py --dataset-name="ifMNIST" --backbone-name="resnet18_fmnist" --C 0.001 0.01 0.1 1.0 10.0 --n-samples=100 --out-dir=concept_bank/fMNIST --device cpu

