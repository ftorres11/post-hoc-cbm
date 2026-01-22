# ResNet 18 - RIVAL10
python train_pcbm.py --concept-bank="concept_bank/RIVAL10/seed_44/RIVAL10_resnet50_0.1_100.pkl" --dataset="RIVAL10" --backbone-name="resnet50" --out-dir=Experiments/RIVAL10/seed_44/R50_01_100 --lam=2e-4 --device cuda --seed 44
python train_pcbm.py --concept-bank="concept_bank/RIVAL10/seed_64/RIVAL10_resnet50_0.1_100.pkl" --dataset="RIVAL10" --backbone-name="resnet50" --out-dir=Experiments/RIVAL10/seed_64/R50_01_100 --lam=2e-4 --device cuda --seed 64
python train_pcbm.py --concept-bank="concept_bank/RIVAL10/seed_256/RIVAL10_resnet50_0.1_100.pkl" --dataset="RIVAL10" --backbone-name="resnet50" --out-dir=Experiments/RIVAL10/seed_256/R50_01_100 --lam=2e-4 --device cuda --seed 256
python train_pcbm.py --concept-bank="concept_bank/RIVAL10/seed_512/RIVAL10_resnet50_0.1_100.pkl" --dataset="RIVAL10" --backbone-name="resnet50" --out-dir=Experiments/RIVAL10/seed_512/R50_01_100 --lam=2e-4 --device cuda --seed 512
python train_pcbm.py --concept-bank="concept_bank/RIVAL10/seed_1024/RIVAL10_resnet50_0.1_100.pkl" --dataset="RIVAL10" --backbone-name="resnet50" --out-dir=Experiments/RIVAL10/seed_1024/R50_01_100 --lam=2e-4 --device cuda --seed 1024
