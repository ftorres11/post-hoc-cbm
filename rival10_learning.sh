# Pretrained RIVAL10 ResNet
python3 learn_concepts_dataset.py --dataset-name="RIVAL10" --backbone-name="resnet50" --C 0.1 --n-samples=100 --out-dir="concept_bank/RIVAL10/seed_44" --seed 44 --device="cuda"
python3 learn_concepts_dataset.py --dataset-name="RIVAL10" --backbone-name="resnet50" --C 0.1 --n-samples=100 --out-dir="concept_bank/RIVAL10/seed_64" --seed 64 --device="cuda"
python3 learn_concepts_dataset.py --dataset-name="RIVAL10" --backbone-name="resnet50" --C 0.1 --n-samples=100 --out-dir="concept_bank/RIVAL10/seed_256" --seed 256 --device="cuda"
python3 learn_concepts_dataset.py --dataset-name="RIVAL10" --backbone-name="resnet50" --C 0.1 --n-samples=100 --out-dir="concept_bank/RIVAL10/seed_512" --seed 512 --device="cuda"
python3 learn_concepts_dataset.py --dataset-name="RIVAL10" --backbone-name="resnet50" --C 0.1 --n-samples=100 --out-dir="concept_bank/RIVAL10/seed_1024" --seed 1024 --device="cuda"
