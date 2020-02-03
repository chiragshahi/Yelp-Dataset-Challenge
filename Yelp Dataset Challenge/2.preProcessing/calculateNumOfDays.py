from datetime import date;

# user number of days passed since user has been yelping

def calculateNumOfDays(str):
	arr = str.split('-');
	#print(arr);
	currdate =  date.today();
	userJoiningDate = date(int(arr[0]), int(arr[1]), int(arr[2]));
	numOfdays = currdate - userJoiningDate;
	return numOfdays.days
