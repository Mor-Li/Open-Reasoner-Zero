python analysis/atc_data_generate/atc.py --language both --needle_counts 2,4,8,16,32,64,128,256,512 --train_repeats 10 --eval_repeats 5


python analysis/atc_data_generate/atc.py --language both --needle_counts 2,4,8,16,32,64,128,256 --train_repeats 10 --eval_repeats 5 --output_dir data/atc_data_256/

python analysis/atc_data_generate/atc.py --language both --needle_counts 2,4,8,16,32,64,128 --train_repeats 10 --eval_repeats 5 --output_dir data/atc_data_128/

python analysis/atc_data_generate/atc.py --language both --needle_counts 2,4,8,16,32,64 --train_repeats 10 --eval_repeats 5 --output_dir data/atc_data_64/

python analysis/atc_data_generate/atc.py --language both --needle_counts 2,4,8,16,32 --train_repeats 10 --eval_repeats 5 --output_dir data/atc_data_32/


