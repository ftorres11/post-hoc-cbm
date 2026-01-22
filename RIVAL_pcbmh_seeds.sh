# ResNet18 RIVAL10
pcbm_path="Experiments/RIVAL10/seed_44/R50_01_100/pcbm_RIVAL10__resnet50__RIVAL10_resnet50_0__lam:0.0002__alpha:0.99__seed:44.ckpt"
python train_pcbm_h.py --concept-bank="concept_bank/RIVAL10/seed_44/RIVAL10_resnet50_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/RIVAL10/seed_44/R50_01_100 --dataset="RIVAL10" --device cuda --backbone_name resnet50
pcbm_path="Experiments/RIVAL10/seed_64/R50_01_100/pcbm_RIVAL10__resnet50__RIVAL10_resnet50_0__lam:0.0002__alpha:0.99__seed:64.ckpt"
python train_pcbm_h.py --concept-bank="concept_bank/RIVAL10/seed_64/RIVAL10_resnet50_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/RIVAL10/seed_64/R50_01_100 --dataset="RIVAL10" --device cuda --backbone_name resnet50
pcbm_path="Experiments/RIVAL10/seed_256/R50_01_100/pcbm_RIVAL10__resnet50__RIVAL10_resnet50_0__lam:0.0002__alpha:0.99__seed:256.ckpt"
python train_pcbm_h.py --concept-bank="concept_bank/RIVAL10/seed_256/RIVAL10_resnet50_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/RIVAL10/seed_256/R50_01_100 --dataset="RIVAL10" --device cuda --backbone_name resnet50
pcbm_path="Experiments/RIVAL10/seed_512/R50_01_100/pcbm_RIVAL10__resnet50__RIVAL10_resnet50_0__lam:0.0002__alpha:0.99__seed:512.ckpt"
python train_pcbm_h.py --concept-bank="concept_bank/RIVAL10/seed_512/RIVAL10_resnet50_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/RIVAL10/seed_512/R50_01_100 --dataset="RIVAL10" --device cuda --backbone_name resnet50
pcbm_path="Experiments/RIVAL10/seed_1024/R50_01_100/pcbm_RIVAL10__resnet50__RIVAL10_resnet50_0__lam:0.0002__alpha:0.99__seed:1024.ckpt"
python train_pcbm_h.py --concept-bank="concept_bank/RIVAL10/seed_1024/RIVAL10_resnet50_0.1_100.pkl" --pcbm-path=$pcbm_path --out-dir=Experiments/RIVAL10/seed_1024/R50_01_100 --dataset="RIVAL10" --device cuda --backbone_name resnet50

