## Trained ResNet 18
# Incomplete fMNIST
python3 learn_concepts_dataset.py --dataset-name="ifMNIST" --backbone-name="resnet18_fmnist" --C 0.1 --n-samples=100 --out-dir="concept_bank/seed_44/fMNIST" --seed 44 --device="cpu"
python3 learn_concepts_dataset.py --dataset-name="ifMNIST" --backbone-name="resnet18_fmnist" --C 0.1 --n-samples=100 --out-dir="concept_bank/seed_64/fMNIST" --seed 64 --device="cpu"
python3 learn_concepts_dataset.py --dataset-name="ifMNIST" --backbone-name="resnet18_fmnist" --C 0.1 --n-samples=100 --out-dir="concept_bank/seed_256/fMNIST" --seed 256 --device="cpu"
python3 learn_concepts_dataset.py --dataset-name="ifMNIST" --backbone-name="resnet18_fmnist" --C 0.1 --n-samples=100 --out-dir="concept_bank/seed_512/fMNIST" --seed 512 --device="cpu"
python3 learn_concepts_dataset.py --dataset-name="ifMNIST" --backbone-name="resnet18_fmnist" --C 0.1 --n-samples=100 --out-dir="concept_bank/seed_1024/fMNIST" --seed 1024 --device="cpu"
# Complete fMNIST
python3 learn_concepts_dataset.py --dataset-name="cfMNIST" --backbone-name="resnet18_fmnist" --C 0.1 --n-samples=100 --out-dir="concept_bank/seed_44/fMNIST" --seed 44 --device="cpu"
python3 learn_concepts_dataset.py --dataset-name="cfMNIST" --backbone-name="resnet18_fmnist" --C 0.1 --n-samples=100 --out-dir="concept_bank/seed_64/fMNIST" --seed 64 --device="cpu"
python3 learn_concepts_dataset.py --dataset-name="cfMNIST" --backbone-name="resnet18_fmnist" --C 0.1 --n-samples=100 --out-dir="concept_bank/seed_256/fMNIST" --seed 256 --device="cpu"
python3 learn_concepts_dataset.py --dataset-name="cfMNIST" --backbone-name="resnet18_fmnist" --C 0.1 --n-samples=100 --out-dir="concept_bank/seed_512/fMNIST" --seed 512 --device="cpu"
python3 learn_concepts_dataset.py --dataset-name="cfMNIST" --backbone-name="resnet18_fmnist" --C 0.1 --n-samples=100 --out-dir="concept_bank/seed_1024/fMNIST" --seed 1024 --device="cpu"

