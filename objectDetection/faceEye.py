##HAAR cascade classifier，先把positive image(圖像中有我們需要的物體，例:人臉)和negative image分開來，
##用一些方法提取features，這時候可能features有很多，假設100000個，這時可使用adaboost演算法，他會用弱的分類器來訓練強的分類器
##這時候features可能減少到5000個，但還是很多，所以cascade classifier的概念就出現了，
##假設有一個block在圖像中滑動，當該block出現臉的特徵數太少時，之後分類就直接跳過該方格，就可以極大的加快效能