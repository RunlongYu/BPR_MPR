# BPR_Bayesian-Personalized-Ranking_MPR_Multiple-Pairwise-Ranking

- Implement Steffen Rendle, et al. Bayesian personalized ranking from implicit feedback (run `BPR.py`);  
- Implement Runlong Yu, et al. Multiple Pairwise Ranking with Implicit Feedback (run `MPR.py`) in Python3.

## Dataset describe: MovieLens100K; 943 Users; 1682 Items

### More details about MovieLens datasets at https://grouplens.org/datasets/movielens .  
### Our code does not depend on the datasets. Just split the dataset you are interested in into the training set and test set, replace the `train.txt` and `test.txt` files, you can run BPR and MPR easily.

## About MPR Framework:

![avatar](https://www.researchgate.net/profile/Runlong_Yu/publication/329800227/figure/fig1/AS:710463142785025@1546399021268/Different-data-divisions-between-BPR-left-and-MPR-right_W640.jpg)

![avatar](https://www.researchgate.net/profile/Runlong_Yu/publication/328436286/figure/fig1/AS:684703258521600@1540257386264/Illustration-of-preference-assumption_W640.jpg)

More details about MPR see our paper or poster at https://www.researchgate.net/profile/Runlong_Yu .

---

### Note:  
In the MPR paper, the dataset is divided into different sets according to popularity, and the method is too dependent on the dataset. In the open source MPR code, we have used the negative sampling method (refer to AoBPR, DNS methods) instead of the method of data division, and achieved better results.

AoBPR at https://dl.acm.org/citation.cfm?id=2556248 .  
DNS at https://dl.acm.org/citation.cfm?id=2484126 .

---

## About Runlong Yu:

LinkedIn: https://www.linkedin.com/in/runlongyu  
Facebook: https://www.facebook.com/YuRunlong

