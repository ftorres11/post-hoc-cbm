pcbm_path="Experiments/CUB/R18_01_100/pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam:0.0002__alpha:0.99__seed:42.ckpt"
python train_pcbm_h.py --concept-bank="concept_bank/cub_resnet18_cub_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/CUB/R18_01_100 --dataset="cub" --device cpu
pcbm_path="Experiments/CUB/R18_001_100/pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam:0.0002__alpha:0.99__seed:42.ckpt"
python train_pcbm_h.py --concept-bank="concept_bank/cub_resnet18_cub_0.01_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/CUB/R18_001_100 --dataset="cub" --device cpu
pcbm_path="Experiments/CUB/R18_0001_100/pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam:0.0002__alpha:0.99__seed:42.ckpt"
python train_pcbm_h.py --concept-bank="concept_bank/cub_resnet18_cub_0.001_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/CUB/R18_0001_100 --dataset="cub" --device cpu
pcbm_path="Experiments/CUB/R18_1_100/pcbm_cub__resnet18_cub__cub_resnet18_cub_1__lam:0.0002__alpha:0.99__seed:42.ckpt"
python train_pcbm_h.py --concept-bank="concept_bank/cub_resnet18_cub_1.0_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/CUB/R18_1_100 --dataset="cub" --device cpu
pcbm_path="Experiments/CUB/R18_10_100/pcbm_cub__resnet18_cub__cub_resnet18_cub_10__lam:0.0002__alpha:0.99__seed:42.ckpt"
python train_pcbm_h.py --concept-bank="concept_bank/cub_resnet18_cub_10.0_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/CUB/R18_10_100 --dataset="cub" --device cpu

