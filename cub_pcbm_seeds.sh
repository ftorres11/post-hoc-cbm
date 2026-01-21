# ResNet 18 - CUB
python train_pcbm.py --concept-bank="concept_bank/CUB/seed_44/cub_resnet18_cub_0.1_100.pkl" --dataset="cub" --backbone-name="resnet18_cub" --out-dir=Experiments/CUB/seed_44/R18_01_100 --lam=2e-4 --device cpu --seed 44
python train_pcbm.py --concept-bank="concept_bank/seed_64/cub_resnet18_cub_0.1_100.pkl" --dataset="cub" --backbone-name="resnet18_cub" --out-dir=Experiments/CUB/seed_64/R18_01_100 --lam=2e-4 --device cpu --seed 64
python train_pcbm.py --concept-bank="concept_bank/seed_256/cub_resnet18_cub_0.1_100.pkl" --dataset="cub" --backbone-name="resnet18_cub" --out-dir=Experiments/CUB/seed_256/R18_01_100 --lam=2e-4 --device cpu --seed 256
python train_pcbm.py --concept-bank="concept_bank/seed_512/cub_resnet18_cub_0.1_100.pkl" --dataset="cub" --backbone-name="resnet18_cub" --out-dir=Experiments/CUB/seed_512/R18_01_100 --lam=2e-4 --device cpu --seed 512
python train_pcbm.py --concept-bank="concept_bank/seed_1024/cub_resnet18_cub_0.1_100.pkl" --dataset="cub" --backbone-name="resnet18_cub" --out-dir=Experiments/CUB/seed_1024/R18_01_100 --lam=2e-4 --device cpu --seed 1024

# ResNet 18 - CREAM
#python train_pcbm.py --concept-bank="concept_bank/CUB/seed_44/cub_resnet18_cream_0.1_100.pkl" --dataset="cub" --backbone-name="resnet18_cream" --out-dir=Experiments/CUB/seed_44/R18_CREAM_01_100 --lam=2e-4 --device cpu --seed 44
#python train_pcbm.py --concept-bank="concept_bank/seed_64/cub_resnet18_cream_0.1_100.pkl" --dataset="cub" --backbone-name="resnet18_cream" --out-dir=Experiments/CUB/seed_64/R18_01_100 --lam=2e-4 --device cpu --seed 64
#python train_pcbm.py --concept-bank="concept_bank/seed_256/cub_resnet18_cream_0.1_100.pkl" --dataset="cub" --backbone-name="resnet18_cream" --out-dir=Experiments/CUB/seed_256/R18_01_100 --lam=2e-4 --device cpu --seed 256
#python train_pcbm.py --concept-bank="concept_bank/seed_512/cub_resnet18_cream_0.1_100.pkl" --dataset="cub" --backbone-name="resnet18_cream" --out-dir=Experiments/CUB/seed_512/R18_01_100 --lam=2e-4 --device cpu --seed 512
#python train_pcbm.py --concept-bank="concept_bank/seed_1024/cub_resnet18_cream_0.1_100.pkl" --dataset="cub" --backbone-name="resnet18_cream" --out-dir=Experiments/CUB/seed_1024/R18_01_100 --lam=2e-4 --device cpu --seed 1024

# Inception v3
#python train_pcbm.py --concept-bank="concept_bank/seed_44/CUB/cub_inceptionv3_concept_0.1_100.pkl" --dataset="cub" --backbone-name="inceptionv3_concept" --out-dir=Experiments/CUB/seed_44/Iv3_01_100 --lam=2e-4 --device cpu --seed 44
#python train_pcbm.py --concept-bank="concept_bank/seed_64/CUB/cub_inceptionv3_concept_0.1_100.pkl" --dataset="cub" --backbone-name="inceptionv3_concept" --out-dir=Experiments/CUB/seed_64/Iv3_01_100 --lam=2e-4 --device cpu --seed 64
#python train_pcbm.py --concept-bank="concept_bank/seed_256/CUB/cub_inceptionv3_concept_0.1_100.pkl" --dataset="cub" --backbone-name="inceptionv3_concept" --out-dir=Experiments/CUB/seed_256/Iv3_01_100 --lam=2e-4 --device cpu --seed 256
#python train_pcbm.py --concept-bank="concept_bank/seed_512/CUB/cub_inceptionv3_concept_0.1_100.pkl" --dataset="cub" --backbone-name="inceptionv3_concept" --out-dir=Experiments/CUB/seed_512/Iv3_01_100 --lam=2e-4 --device cpu --seed 512
#python train_pcbm.py --concept-bank="concept_bank/seed_1024/CUB/cub_inceptionv3_concept_0.1_100.pkl" --dataset="cub" --backbone-name="inceptionv3_concept" --out-dir=Experiments/CUB/seed_1024/Iv3_01_100 --lam=2e-4 --device cpu --seed 1024
