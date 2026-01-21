pcbm_path="Experiments/fMNIST/seed_44/R18_01_100/pcbm_cfMNIST__resnet18_fmnist__cfMNIST_resnet18_fmnist_0__lam:0.0002__alpha:0.99__seed:44.ckpt"
python train_pcbm_h.py --concept-bank="concept_bank/seed_44/fMNIST/cfMNIST_resnet18_fmnist_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/fMNIST/seed_44/R18_01_100  --device cpu --backbone_name resnet18_fmnist --dataset fMNIST
pcbm_path="Experiments/fMNIST/seed_64/R18_01_100/pcbm_cfMNIST__resnet18_fmnist__cfMNIST_resnet18_fmnist_0__lam:0.0002__alpha:0.99__seed:64.ckpt"
python train_pcbm_h.py --concept-bank="concept_bank/seed_64/fMNIST/cfMNIST_resnet18_fmnist_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/fMNIST/seed_64/R18_01_100  --device cpu --backbone_name resnet18_fmnist --dataset fMNIST
pcbm_path="Experiments/fMNIST/seed_256/R18_01_100/pcbm_cfMNIST__resnet18_fmnist__cfMNIST_resnet18_fmnist_0__lam:0.0002__alpha:0.99__seed:256.ckpt"
python train_pcbm_h.py --concept-bank="concept_bank/seed_256/fMNIST/cfMNIST_resnet18_fmnist_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/fMNIST/seed_256/R18_01_100  --device cpu --backbone_name resnet18_fmnist --dataset fMNIST
pcbm_path="Experiments/fMNIST/seed_512/R18_01_100/pcbm_cfMNIST__resnet18_fmnist__cfMNIST_resnet18_fmnist_0__lam:0.0002__alpha:0.99__seed:512.ckpt"
python train_pcbm_h.py --concept-bank="concept_bank/seed_512/fMNIST/cfMNIST_resnet18_fmnist_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/fMNIST/seed_512/R18_01_100  --device cpu --backbone_name resnet18_fmnist --dataset fMNIST
pcbm_path="Experiments/fMNIST/seed_1024/R18_01_100/pcbm_cfMNIST__resnet18_fmnist__cfMNIST_resnet18_fmnist_0__lam:0.0002__alpha:0.99__seed:1024.ckpt"
python train_pcbm_h.py --concept-bank="concept_bank/seed_1024/fMNIST/cfMNIST_resnet18_fmnist_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/fMNIST/seed_1024/R18_01_100  --device cpu --backbone_name resnet18_fmnist --dataset fMNIST

