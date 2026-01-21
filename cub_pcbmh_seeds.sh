# ResNet18 CUB
pcbm_path="Experiments/CUB/seed_44/R18_01_100/pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam:0.0002__alpha:0.99__seed:44.ckpt"
python train_pcbm_h.py --concept-bank="concept_bank/CUB/seed_44/cub_resnet18_cub_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/CUB/R18_01_100 --dataset="cub" --device cpu

#pcbm_path="Experiments/CUB/seed_64/R18_01_100/pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam:0.0002__alpha:0.99__seed:64.ckpt"
#python train_pcbm_h.py --concept-bank="concept_bank/seed_64/cub_resnet18_cub_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/CUB/R18_01_100 --dataset="cub" --device cpu

#pcbm_path="Experiments/CUB/seed_256/R18_01_100/pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam:0.0002__alpha:0.99__seed:256.ckpt"
#python train_pcbm_h.py --concept-bank="concept_bank/seed_256/cub_resnet18_cub_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/CUB/R18_01_100 --dataset="cub" --device cpu

#pcbm_path="Experiments/CUB/seed_512/R18_01_100/pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam:0.0002__alpha:0.99__seed:512.ckpt"
#python train_pcbm_h.py --concept-bank="concept_bank/seed_512/cub_resnet18_cub_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/CUB/R18_01_100 --dataset="cub" --device cpu

#pcbm_path="Experiments/CUB/seed_1024/R18_01_100/pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam:0.0002__alpha:0.99__seed:1024.ckpt"
#python train_pcbm_h.py --concept-bank="concept_bank/seed_1024/cub_resnet18_cub_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/CUB/R18_01_100 --dataset="cub" --device cpu

# ResNet18 - CREAM
#pcbm_path="Experiments/CUB/seed_44/R18_CREAM_01_100/pcbm_cub__resnet18_cream__cub_resnet18_cream_0__lam:0.0002__alpha:0.99__seed:44.ckpt"
#python train_pcbm_h.py --concept-bank="concept_bank/CUB/seed_44/cub_resnet18_cream_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/CUB/R18_CREAM_01_100 --dataset="cub" --device cpu --backbone_name resnet18_cream

#pcbm_path="Experiments/CUB/seed_64/R18_CREAM_01_100/pcbm_cub__resnet18_cream__cub_resnet18_cream_0__lam:0.0002__alpha:0.99__seed:64.ckpt"
#python train_pcbm_h.py --concept-bank="concept_bank/seed_64/cub_resnet18_cream_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/CUB/R18_CREAM_01_100 --dataset="cub" --device cpu

#pcbm_path="Experiments/CUB/seed_256/R18_CREAM_01_100/pcbm_cub__resnet18_cream__cub_resnet18_cream_0__lam:0.0002__alpha:0.99__seed:256.ckpt"
#python train_pcbm_h.py --concept-bank="concept_bank/seed_256/cub_resnet18_cream_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/CUB/R18_CREAM_01_100 --dataset="cub" --device cpu

#pcbm_path="Experiments/CUB/seed_512/R18_CREAM_01_100/pcbm_cub__resnet18_cream__cub_resnet18_cream_0__lam:0.0002__alpha:0.99__seed:512.ckpt"
#python train_pcbm_h.py --concept-bank="concept_bank/seed_512/cub_resnet18_cream_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/CUB/R18_CREAM_01_100 --dataset="cub" --device cpu

#pcbm_path="Experiments/CUB/seed_1024/R18_CREAM_01_100/pcbm_cub__resnet18_cream__cub_resnet18_cream_0__lam:0.0002__alpha:0.99__seed:1024.ckpt"
#python train_pcbm_h.py --concept-bank="concept_bank/seed_1024/cub_resnet18_cream_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/CUB/R18_CREAM_01_100 --dataset="cub" --device cpu

# Inception v3 concept
#pcbm_path="Experiments/CUB/seed_44/Iv3_01_100/pcbm_cub__inceptionv3_concept__cub_inceptionv3_concept_0__lam:0.0002__alpha:0.99__seed:44.ckpt"
#python train_pcbm_h.py --concept-bank="concept_bank/seed_44/CUB/cub_inceptionv3_concept_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/CUB/Iv3_01_100 --dataset="cub" --device cpu --backbone_name inceptionv3_concept

#pcbm_path="Experiments/CUB/seed_64/Iv3_01_100/pcbm_cub__inceptionv3_concept__cub_inceptionv3_concept_0__lam:0.0002__alpha:0.99__seed:64.ckpt"
#python train_pcbm_h.py --concept-bank="concept_bank/seed_64/CUB/cub_inceptionv3_concept_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/CUB/Iv3_01_100 --dataset="cub" --device cpu

#pcbm_path="Experiments/CUB/seed_256/Iv3_01_100/pcbm_cub__inceptionv3_concept__cub_inceptionv3_concept_0__lam:0.0002__alpha:0.99__seed:256.ckpt"
#python train_pcbm_h.py --concept-bank="concept_bank/seed_256/CUB/cub_inceptionv3_concept_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/CUB/Iv3_01_100 --dataset="cub" --device cpu

#pcbm_path="Experiments/CUB/seed_512/Iv3_01_100/pcbm_cub__inceptionv3_concept__cub_inceptionv3_concept_0__lam:0.0002__alpha:0.99__seed:512.ckpt"
#python train_pcbm_h.py --concept-bank="concept_bank/seed_512/CUB/cub_inceptionv3_concept_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/CUB/Iv3_01_100 --dataset="cub" --device cpu

#pcbm_path="Experiments/CUB/seed_1024/Iv3_01_100/pcbm_cub__inceptionv3_concept__cub_inceptionv3_concept_0__lam:0.0002__alpha:0.99__seed:1024.ckpt"
#python train_pcbm_h.py --concept-bank="concept_bank/seed_1024/CUB/cub_inceptionv3_concept_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/CUB/Iv3_01_100 --dataset="cub" --device cpu

