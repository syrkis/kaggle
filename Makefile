download:
	kaggle competitions download -c $(comp)
	unzip $(comp).zip -d data
	rm $(comp).zip
	
train:
	python main.py --comp=$(comp)

submit:
	python main.py --comp=$(comp)
	kaggle competitions submit -c $(comp) -f data/submission.csv
	rm data/submission.csv


