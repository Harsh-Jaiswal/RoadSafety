import pandas as pd
from datetime import datetime

date_format = "%Y-%m-%d"



df = pd.read_excel('Data/cleaned_data_2.xlsx')


def total_hits(area_name, date1 , date2):

    t1 = datetime.strptime(date1, date_format)
    t2 = datetime.strptime(date2, date_format)
    print(t1)
    print(t2)
    i=0
    count=0

    while (df.iloc[i][0] != area_name):
        if (i < 142417):
            i = i + 1
            #print(i)
        else:
            return 0

    while (df.iloc[i][0] == area_name):
        if (df.iloc[i][2] > t1 and df.iloc[i][2] < t2):
            count=count+1

        if (i < 142417):
            i = i + 1
        else:
            return count

    return count



print(total_hits( "A Narayanapura", "2018-02-27" , "2018-04-02"))

#///2018-04-14/2019-09-11