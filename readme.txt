Code for Kaggle`s StateFarm distracted driver classification
In: image
Out: one of 9 classes: Driver is driving well , Driver is using cellphone in right hand , Driver is drinking , etc

see blog for results:
http://onsoftwaredev.blogspot.co.il/2016/07/statefarm-experiment-1.html
http://onsoftwaredev.blogspot.co.il/2016/07/confusionmatrix.html
http://onsoftwaredev.blogspot.co.il/2016/07/statefarm-experiment-2.html
http://onsoftwaredev.blogspot.co.il/2016/08/statefarm-experiment-3-vgg16-finetune.html
http://onsoftwaredev.blogspot.co.il/2016/08/statefarm-retrospect.html

Code:   (this was my first real project with keras/python, so....)
statefram-googlenet-finetune.ipynb  - finetuning googlenet, note that it has 3 heads (2 aux), so it's a bit tricky.
statefarm-baseline-fromscratch.ipynb - baseline

python code:
keras/finetune -  folder contains python utilities for loading/finetuning networks, including locking layers, persistening results of locked layers to disk.
		  (Lately keras added API to do it automatically, so use thier API)
	          I used a googletnet implementation (not my own)


