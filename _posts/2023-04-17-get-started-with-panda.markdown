---
layout: post
# layout: single
title:  "Get started with pands"
date:   2023-04-17 12:51:28 -0800
categories: jekyll update
---

# Panda

 * [https://lab.rtscloud.in/#/](https://lab.rtscloud.in/#/)

 ```


import panda as pd


iris.columns

iris.head(25)
iris.tail(25)

groupeddata = iris.groupby('variety')  # GRoupby one (or more than 1 column)
for groupname, data in groupeddata:
	print(groupname)                   # SETOSA, VIGINICA, 
	print(data)


# Aggregate function
# Variety is preserved (beginning of row)
#  Label of columns label1: label2 or list_of_labels_2
groupeddata.agg({"petal.length":"min", "petal.width":["min","max","median"]})

jsondata = pd.read_json("https://jsonplaceholder.typicode.com/todos")
jsondata       
# returns table with index_number, userId, Id, title, completed

# Save a dataframe in many different formats
jsondata.to_<tab>       
jsondata.to_csv("C://data//jsondata.csv", index=False)
jsondata.to_xml("C://data//jsondata.xml", index=False)


# NaN = Not a number, but NaN of of type 'float' ==> type of column = float!

# Join
student=pd.read_excell("C://data/student.xlsx")
dept = pd.read_excel("C://data/dept.xlsx")
student.head()          # rollno, student_name, dept_ident
dept.head()             # dept_id, dept_name
# Inner join
# Match value left_on and right_on (even if slightly differnet types, i.e. float vs integer)
student = pd.merge(student,dept, left_on="dept_ident", right_on="dept_id")

# Left join
# Everything from left table comes including entries with no departments!
studept = pd.merge(student,dept, left_on="dept_ident", right_on="dept_id", how='left')
# Right join
# Even if there are no students in a department, the department will show (with NaN)
studept = pd.merge(student,dept, left_on="dept_ident", right_on="dept_id", how='right')
# Outer join
# = right and left join 
studept = pd.merge(student,dept, left_on="dept_ident", right_on="dept_id", how='outer')


# Drop a column + drop a column in place!
df = iris.drop('new_variety', axis=1)             # iris is not changed
iris.drop('new_variety', axis=1, inplace=True)    # iris is changed

help(iris.drop)
 ```
 * [https://github.com/guipsamora/pandas_exercises/blob/master/01_Getting_%26_Knowing_Your_Data/Occupation/Exercises.ipynb](https://github.com/guipsamora/pandas_exercises/blob/master/01_Getting_%26_Knowing_Your_Data/Occupation/Exercises.ipynb)

 ```
users = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user', sep='|', index_col='user_id')
users = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user', sep='|').set_index('user_id')

# Number of observations
users.shape[0]

# Number of columns + names
users.shape[1]
users.columns

user.index

uesrs.dtypes

# Select columns
users.occupation
users['occupation']

# Unique occupations
users.occupation.nunique()

# Occupation ordered frequency + ...
users.occupation.value_counts()
users.occupation.value_counts().count()     # Unique occupations
users.occupation.value_counts().head()      # Top 5
users.age.value_counts().tail()             # Bottom 5: 7, 10, 11, 66 and 73 years -> only 1 occurrence

# Statistics on columns
users.describe()                   # count, mean, std, min, 25%, 50%, 75%, max (numeric columns only)
users.describe(include='all')      # count, unique, top, freq, mean, .... (all columns)
users.occupation.describe()        # count, unique, top, freq

round(users.age.mean())

 ```

 ```
 c = chipo.groupby('item_name')
c = c.sum()
c = c.sort_values(['quantity'], ascending=False)
c.head(1)
 ```

 ```
# Change the 
iris.columns = iris.columns.str.upper()

# Lambda function on a column
dollarizer = lambda x: float(x[1:-1])
chipo.item_price = chipo.item_price.apply(dollarizer)
 ```

{% include links/all.md %}
