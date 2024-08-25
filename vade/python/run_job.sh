#python -u train_ae.py --epochs 100
python -u train_vade.py --epochs 460 --load_weights --eval_every 20 --alpha 35
#2>&1 | tee -a ./log.txt
