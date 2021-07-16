#3 CUDA_VISIBLE_DEVICES=3 python main_informer.py --model informer --data aiops --attn prob --freq t > out.log
CUDA_VISIBLE_DEVICES=3 python main_aiops.py --model informer --data aiops --attn prob --freq t --freeze true
