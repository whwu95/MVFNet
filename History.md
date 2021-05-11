## CMD

> Train
```
bash scripts/dist_train_recognizer.sh config 8
```

> Test
```
bash scripts/dist_test_recognizer.sh config ckpt 8 [--fcn_testing]
```
score save in default.pkl

> score fusion
```
python report_accuracy.py --scores s1.pkl s2.pkl  --coefficients 1 1 --datalist list.txt
```

> count FLOPs
```
python count_flops.py config
```