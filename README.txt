Pre-work required :
1. Please save following files in any location:
	a. article_user_recommender_bots.py - python script
	b. article_user_recommender_bots.ipynb - ipython notebook
	c. news_articles.csv - Article data
	d. clickstream.csv - Clickstream user data
	e. stopwords.txt - list of stopwords(generated iteratively from the data)
The programs may take longer time in the first run, as it will install packages, and from the second run it will take on an average 15-16 minutes.

Following are the ways in which code can be run:


1. Run on jupytor Notebook(use .pynb extension of code)
	a. Just apply run all option from cell tab.
	b.It will install all packages which is required for the program, whichever packages shows error or warning then please install it using pip command and appropriate version as per python version
	c.The code will request you to enter file location, in that case enter location in quotes as G:/analytics/ISB CBA/Residency/DMG Project - please mind the forwardslash(/) and dont use quotations to enclose the address.
	d.Then the code will request you to enter file name without extension, enter again without quotes as "news_article"
	e.At the end of the run, there are request to enter userid for which you want to view the recommendations, you can give any user id(which would be numeric numbers only like 1,2,3..), it may be the new user id or old, the system will respond accordingly.
	f. You can also view recommendation of all existing user, and you will get only 10 recommendation for any user.

2. Run python script on any command prompt(python installed cmd)-(use .py extension of code)(It does not work on all systems)
	a.Run the code with the following command
	"python scriptname.py"
	b.It will install all packages which is required for the program, whichever packages shows error or warning then please install it using pip command and appropriate version as per python version
	c.The code will request you to enter file location, in that case enter location in quotes as ""G:/analytics/ISB CBA/Residency/DMG Project" - please mind the forwardslash(/).
	d.Then the code will request you to enter file name without extension, enter again in quotes as "news_article"
	e.At the end of the run, there are request to enter userid for which you want to view the recommendations, you can give any user id(which would be numeric numbers only like 1,2,3..), it may be new user id or old, the system will respond accordingly.
	f. You can also view recommendation of all existing user, and you will get only 10 recommendation for any user.