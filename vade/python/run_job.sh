#python -u train_ae.py --epochs 100
python -u train_vade.py --epochs 400 --load_weights 
#2>&1 | tee -a ./log.txt
