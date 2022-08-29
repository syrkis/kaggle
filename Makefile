download:
	kaggle competitions download -c $(comp)
	unzip $(comp).zip -d data
	rm $(comp).zip
	
train:
	python main.py --comp=$(comp)

submit:
	kaggle competitions submit -c $(comp) -f data/submission.csv -m "my kaggle submission"
	rm data/submission.csv


