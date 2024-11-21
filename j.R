Session 1 
Shortcut keys
# ctrl + shift + c (comment and uncomment)

# gdsgsd
# gdgsgs
# gadgag

# ctrl + enter for running the code

if(FALSE)
{
  "logical comment as this will never execute 
  because control can enter into this"
}

11+2


#ctrl + L to clean the console


a=5
b=2
c=a + b
c


name = "r"
name
Name

name = "r"
name
# print is a language construct which takes only single parameter
print(name)



paste("hello",name)

# print is a language construct which takes only single parameter
print(name)
# paste is used for printing multiple params/args
# has default separator adds single space
paste("hello",name)
#removes the space after hello
paste0("hello", name)

#shortform on concatenation
cat("hello", name, a ,b, c)

# formatted string
sprintf("hello",name)
sprintf("hello %s",name)
sprintf("hello %s %d",name,c)


print(paste("hello this is paste()",print("this is print")))

# = <- -> are assignment operators
email <- readline("enter email: ")
email

pwd <- readline(prompt = "enter password : ")
pwd

#datatype 
num1 = as.numeric(readline("num1 : "))
num2 = as.integer(readline("num2 : "))

num1+num2

typeof(num1)

#new line
print("line1 \n line2 \n line3")
writeLines("line1 \n line2 \n line3")

x = 1 
while(x<10){
  print(x)
  x = x+1
}


x = 1 
while(x<10){
  x = x+1
  if(x ==5)
    next #used to skip one or multiple iteration in the loops
  print(x)
}


x = 1 
while(x<10){
  x = x+1
  if(x ==5)
    #next #used to skip one or multiple iteration in the loops
    break
  print(x)
}


Session 2
Vectors

1:6
a = 5L
is.atomic(a)
is.vector(a)
is.integer(a)
is.double(a)

# 1. using atomic values like 11 5 7
# 2. using range operator like 1:5
# 3. c()
# 4. seq()
# 5. unlisting the list
# 6. using repeat function rep(value,length)
# 7. using random values

# r will do implicit type casting and make it similar type
v = c(11,22,5,13,33,c("ps"))
v
is.vector(v)
is.atomic(v)

typeof(v) # to check the datatype of object
class(v) # to get the class type of the object

v = c(1:5,15:20)
v

#seq()
v = seq(1,20)
v
#by parameter of seq() decides the step size
v = seq(1,20,by=3)
V

#length.out(value) decides length of the data structure
#object or it decides no of components to be generated from
# the given range

v = seq(1,20,length.out=5)
v = seq(1,5,length.out=20)
v = seq(10,by=0.2,length.out=20)
v

a = c(11,22,33)
b = seq(1:3)
c = 21:23

v1 = c(a,b,c)
v1

#type conversion - lower to higher 
#type casting - higher to lower

#naming indexes
#in R indexing starts with 1 and not 0


#another way of creating indexes
nv = c(101,102,103)
setNames(nv,c("R","python","android"))
nv[1]

# perform operations on vectors
# u can perform arithmetic operation on same size and same 
# type of vectors
a = c(5,9,13)
b = c(23,14,7)

#a + b
#a - b
#a * b
a == b
a > b
a < b

a = c(5,9,13,14,7)
b = c(23,14,7)
a %in% b #checks if the values of a are in b

sort(a)
sort(a, decreasing = T) #T/F or TRUE or FALSE is used in R and not true/false

#create an empty vector
ev = c()
ev 
#if something is null it means it is created but has no values/ memory assigned
ev = vector(length = 7) #2nd way 
ev
is.vector(ev)
length(ev)
# vector declaration with memory pre-allocation
ev = rep(NULL, 7)
ev = rep(NA, 7)
ev = rep(Inf, 7)
ev
length(ev)


v = c(1,2,3)
#deleting the vector
v = NULL #null is assigned to data object to delete it
v

ev = rep(NA, 7)
length(ev)
for(i in 1:7){
  ev[i]=i
}
ev

#creating vector with random values'
# 1. sample()
set.seed(3)
sample(101:107,size =15, replace = T)
sample(101:107,size =15, replace = T) #generates random numbers from the range 
#sample(101:107,size =15, replace = F) #replace = F doesn't allow duplicate values hence we get an error
# 2. rnorm()
rnorm(5)
rnorm(5,mean = 2)
rnorm(5,mean = 2, sd = 2)
set.seed(3) #doesn't change the random values each time it runs... set.seed can have any random value
rnorm(10)

# 3. runif()
set.seed(3)
runif(7,min =10,max =15)

#using in built vectors
#lett
#letters
#LETTERS

v = c(11,22,13)
v[4]=55 # appending the vector
v
#modify or update the vector
v[2] = 77
v
#v[2] = NULL # error -  we cannot delete value from vector only add and modify

#delete the vector
#rm(v)
#v
#delete the vector by assigning 0 value
#v = 0
#v
v = 15:23
v
v = v[-2]
v
#accessing the vectors
v
v[2]
v[-2] #show all except second
v[2:4] #range as index
v[c(2,4,6)] #vector defined with c() as indexing
v[seq(2,5)] #vector defined with seq() as indexing
v[c(T,F)]
v
v[c(T,F,F,F)]

temp = c(23,5,21,13,34,45,43)
month = c("jan","feb","mar","apr","may","june","july")
month[temp<20]
month[temp<20 | temp > 40]
Session 3
Matrix
#create matrix
mt = matrix(seq(10,160,by = 10),ncol = 4, byrow = T)
mt

#access the element
mt[3,2]

#Create a matrix with values from 10 to 160, filled by columns
mt = matrix(seq(10, 160, by = 10), bycol = TRUE)
mt
#This line has an error because bycol is not a valid argument. It should be byrow = FALSE if you want to fill by columns.

#Create a 5x5 matrix
mt = matrix(seq(10, 160, by = 10), ncol=5,nrow=5)
mt

#Create a 5x5 matrix, filled by rows
mt = matrix(seq(10, 160, by = 10), ncol=5,byrow=T)
mt

#Create a 4x5 matrix with custom row and column names
mt = matrix(seq(10,160, by =10), ncol =5 , byrow = T, dimnames = list(letters[1:4],LETTERS[1:5]))
mt

Using rbind and cbind
#Append a row to the matrix
r = c(100, 200, 300, 400, 500)
mt = rbind(mt,r)
mt


#append col to the matrix
r = c(100, 200, 300, 400)
mt = cbind(mt,r)
mt

#Access different elements and ranges:
mt[1:5]    # Accesses the first 5 elements in column-major order.
mt[5:1]    # Accesses elements 5 to 1 in column-major order.
mt[4:3]    # This syntax is incorrect for accessing a range in a matrix.
mt[-2]     # Accesses all elements except the second in column-major order.
mt[1]      # Accesses the first element in column-major order.
mt


#Access specific rows and columns
mt[2, ]           # Shows the second row and all columns.
mt[2, c(2, 3)]    # Shows the second row and the 2nd and 3rd columns.
mt[, 3]           # Shows only the third column.

#remove rows and columns
mt
mt = mt[-5, ]     # Removes the 5th row.
mt
mt = mt[, -5]     # Removes the 5th column.
mt
mt = mt[c(-1, -2)] # This line has an error. It should be mt = mt[-c(1, 2), ] to remove the 1st and 2nd rows.
mt


#Reset matrix with row and column names:
mt = matrix(seq(10, 160, by = 10), ncol = 5, byrow = TRUE, dimnames = list(letters[1:4], LETTERS[1:5]))
mt

rownames(mt) = NULL
mt
colnames(mt) = NULL
mt

rownames(mt) = c("ps", "pps", "prs", "lpl")
mt
colnames(mt) = c("xzy", "abc", "dfc", "scd", "ssd")
Mt

#Create a 3-dimensional array
result = array(seq(10,480, by=10),dim = c(4,4,3),dimnames=list(c("s1","s2","s3","s4"),c("java","c","cpp","r"),c("mba","mca","mim")))
result

#Appending rows to the array
r = c("101", "201", "501", "334")
result = rbind(result, r)
result

result = rbind(r, result)
result


#assignment
{
  num1 <- as.integer(readline(prompt = "Enter the start of the sequence: "))
  num2 <- as.integer(readline(prompt = "Enter the end of the sequence: "))
  by1 <- as.integer(readline(prompt = "Enter the increment (by) value: "))
  col <- as.integer(readline(prompt = "Enter the number of columns: "))
  row <- as.integer(readline(prompt = "Enter the number of rows: "))
  mt=matrix(seq(num1,num2,by=by1),ncol = col,nrow=row)
  mt
}

Session 4 
Lists
#list - treats all as diff vector also keeps the types as it is
rollno = 1:5
names = letters[1:5]
course = c("mca","mba","bba","mcom","ma")

rec = list(rollno, names, course)
rec
typeof(rec)

#vector - treats all the same of the highest type (here - character) and all are written in same line
vrec = c(rollno,names,course)
vrec
typeof(vrec)



#accessing list via indexing
rec[[3]][1]

Dataframe
#dataframe

mca = data.frame(
  rollno = 1:5,
  names = c("pranjal","karan","pratik","falguni","vijay"),
  city = c("mumbai","mumbai","delhi","pune","pune"),
  stringsAsFactors = F
)
mca
#convert list into dataframe
as.data.frame(rec,col.names = c("rollno","names","courses"))



#Alternative for column names
rec = list(rollno = rollno,name =  names,course =  course)
rec
rec$rollno[1] #$ - used to represent column name


#accessing df
mca[1] #considers column instead of row (single value is considered as col in df)
mca[1,] #when comma is used it considers row - accessing row of df
mca [,2] # accessing col of df 
mca[1,2] #second col - first row
mca[[1]] #column
mca[[2]][1] #second col - first row
mca$rollno
mca$names[1]

#appending cols in df
#1 way
mca
course = c("MBA","MCA","MS","MCA","")
mca["course"] = course
Mca
#2 way
#add dob column to df
dob = as.Date(c("2002-01-07","1999-10-29","2022-04-13","2002-03-04","2002-04-05","2002-03-05"))
mca = cbind(mca,dob)
mca

#appending row in df
#1 way
stud = list(6,"nilesh","delhi","MBA")
mca[6,] = stud
mca


#2 way
#add row to df
stud2 = list(7,"xyz","pune","MCA","2002-03-06")
mca = rbind(mca,stud2)
mca[-7,]
mca

mca <- mca[-c(7), ]








salary = c(1400000,1200000,50000,55000,10000,20000,10000)
mca= cbind(mca,salary)
mca
subset(mca,mca$city %in% c("pune"))
subset(mca,mca$salary == max(mca$salary))
subset(mca,mca$city %in% c("pune") | mca$salary == max(mca$salary))
subset(mca,as.Date(mca$dob) < as.Date("2002-01-07"),select = c(1,4,6))


Session 5
Functions
#functions in R
wish = function(){
  print("hello")
}
wish()


wish = function(name,institue){
  print("hello")
  paste("good morning" , name , "you are from " , institue)
}
wish("Pranjal","KJSIM")


wish = function(name,institue){
  print("hello")
  paste("good morning" , name , "you are from " , institue)
}
wish("Pranjal","KJSIM")
wish(institue = "KJSIM", name = "PS")


#lazy evaluation of a function
wish = function(name,institute,city){
  paste("good morning" , name , "you are from " , institute)
}
wish("Pranjal","KJSIM")


#default parameterized
wish = function(name = "PS"){
  paste(name)
}
wish()  


sum = function(a,b){
  return (a+b);
}
sum(10,20)


area = function(l,h){
  area = (l+h)
  perimeter = (l * h)
  circumference = (2.13 * (l * h))
  result = list(area=area,perimeter=perimeter,circumference=circumference)
  return (result)
}
result = area(5,7)
paste("area: ", result["area"], 
      "perimeter: ", result["perimeter"],
      "circumference: ", result["circumference"]
)
sum(5,6)



pow = function(x,n) x^n
pow(2,3)

Apply
x = seq(1,4)
y = seq(5,8)
z = seq(10,13)
df = data.frame(x,y,z)
df
df = as.matrix(df)
x = matrix(seq(1:16),nrow = 4)
apply(df,1,sum)

x = matrix(seq(1:16),nrow = 4)
apply(x,1,sum)
x


pow = function(x) x^x
apply(x,MARGIN = c(1,2),pow)

apply(x,MARGIN = c(1,2),function(x) x^x)

name1 = matrix(c("pranjal", "ps","pps","abc"),nrow = 2)
name1
apply(name1,1,function(name1)toupper(name1))


apply(name1,1,function(name1)tolower(name1))

typeof(ap)
is.matrix(ap)
is.vector(ap)
is.array(ap)
#when it is by row or column it will return vector, when it is matrix it will return matrix when array it will array
apply(name1,1,function(name1)tolower(name1))


pow = function(x) x^x
apply(x,MARGIN = c(1,2),function(x) x^x)
sapply(x,pow)

pow = function(x) x+x
apply(x,MARGIN = c(1,2),function(x) x*x)
sapply(x,pow)
sapply(x,pow, simplify = T)
lapply(x,pow)



File Handling
#file handling

#set and get temporary working directory
setwd("C:/Users/exam.201STUD63/Documents/R_Pranjal")
getwd()

#list all the files
list.files()

#list directories 
list.dirs()

. is for hidden directories

#file handling

#set and get temporary working directory
setwd("C:/Users/exam.201STUD63/Documents/R_Pranjal")
getwd()

#list all the files
list.files()

#list directories 
list.dirs()
#. is for hidden directories

#to check if file exists
file.exists("abc.txt")
file.exists("functions_session5.R")

#check size of file
file.size("functions_session5.R")
file.size("abc.txt")


#opens the file in appropriate explorer
file.show("test.txt")

#opens the file in R studio in edit mode
file.edit("test.txt")


#checks if the mode is available
file.access("test.txt",6)

#append a file
file.append("test.txt","test.txt")

#if loc and name of file is not known to u 
#opens file select 
file.choose()

#create file
file.create("abc.txt")
list.files()



#copy existing file to a new file
file.copy("test.txt","ab.txt")
#copy existing file to old file
file.copy("test.txt","abc.txt",overwrite = T)

#gets the file info
file.info("test.txt")

#permissions details
file.mode("abc.txt")

#rename the file
file.rename("abc.txt", "s1.txt")

#remove file
file.remove("ab.txt")

#checks if the file is old or new
file_test("-ot", "s1.txt", "test.txt") #older than -ot
file_test("-nt", "s1.txt", "test.txt") #newer than -nt

#read csv file
df = read.csv("data.csv")
df
is.data.frame(df)


ipl = data.frame(
  teamName = c("Mumbai Indians","Chennai Super Kings","Delhi Capitals","Gujrat Titans","Kolkata Knight Riders","Rajasthan Royals","RCB"),
  matchPlayed = c(10,8,8,8,10,5,5),
  matchWon = c(9,3,4,5,8,2,2),
  matchlost = c(1,5,4,3,2,3,3)
)
ipl
write.table(ipl,"ipl.csv", row.names = F, append = T, col.names = F, sep = ",")
read.csv("ipl.csv")


students = data.frame(
  name = c("PS","PPS","Pranjal"),
  age = c(20,22,21)
)
students
write.table(students,"students.csv", row.names = F, append = T, col.names = F, sep = ",")
read.csv("students.csv")

#assignment - 3 csv file -> read all 3 using apply func - list.files pattern - reg exp to list all csv
df = list.files(pattern = ".csv")
apply(df,MARGIN = c(1,2),f)
#create a function() that asks user to perform all the functions


setwd("D:/RAssignment")
d1 = data.frame(name = c("Pranjal","PS", "PSawant"), age = c(21,33,23))
d2 = data.frame(name = c("abc","pqr","xyz"),age = c(20,22,21))
d3 = data.frame(name = c("asd","wer","rwew"),age = c(26,2,41))

write.table(d1,"d1.csv",row.names = F,append = T, col.names = F, sep = ",")
write.table(d2,"d2.csv", row.names = F, append = T, col.names = F, sep = ",")
write.table(d3,"d3.csv", row.names = F, append = T, col.names = F, sep = ",")

d1 <- read.csv("d1.csv", header = FALSE, col.names = c("name", "age"))
d2 <- read.csv("d2.csv", header = FALSE, col.names = c("name", "age"))
d3 <- read.csv("d3.csv", header = FALSE, col.names = c("name", "age"))

combined_df <- rbind(d1, d2, d3)
combined_df

avg = function(){
  mean_age <- apply(combined_df["age"], 2, mean)
  mean_age
}
listall = function(){
  list.files()
}


info = function(){
  var = readline(prompt="Enter file name ");
  var = as.character(var);
  file.info(var);
}

{
  op1 <- readline(prompt="Enter 'a' to calculate age, 'l' to get list of all files 'i' to get information of files: ")
  op <- as.character(op1)
  
  if (op == "a") {
    avg()
  } else if(op == "l") {
    listall()
  } else if(op == "i") {
    info()
  } else {
    cat("Enter correct character")
  }
}

Session 6
XLSX package
#read
install.packages("readxl")
library("readxl")
df = read_xlsx("D:/R Programming/airline.xlsx", sheet = 1, range = "A1:F33")
df 
View(df)






#write
d1 = data.frame(c(1978,22.726,1.703,0.5877,6.108,7.104))
library(writexl)
write_xlsx(d1,"D:/R Programming/airline.xlsx")

JSON package
#read
library("json64")
library("jsonlite")
data = read_json("D:/R Programming/one.json")
data = as.data.frame(data)
data()

data = fromJSON("D:/R Programming/one.json")
df = as.data.frame(data)
df

jsonData = toJSON(df)

df1 = as.data.frame(jsonData)
df1


SQL package
#sql
install.packages("RMySQL")
library("RMySQL")

connection = dbConnect(MySQL(), user = "root", password= "", host= "localhost")

Data visualization
Pie Chart
#data visualization

#1 pie chart

x = c(23,54,19,34,7)
pie(x)

#main -> gives heading to the chart
#labels give label to each field
#colors add colors to field (can also use rainbow func)
pie(x, main = "avg package lpa course-wise", labels = c("mca", "mba","ms","ma","mim"),col = c("lightblue","lightgreen","lightyellow","lightpink","yellow"))



pie(x, main = "avg package lpa course-wise", labels = c("mca", "mba","ms","ma","mim"),col = rainbow(length(x))) 


#legend
x = c(23,34,19,34,7)
pie_per = round(x*100/sum(x),1)
courses = c("mca", "mba","ms","ma","mim")
color = c("lightblue","lightgreen","lightyellow","lightpink","yellow")
png("D:/R Programming/pie.png") #should be before graph always
pie(x,main = "lpa avg package course-wise", labels = pie_per,col = color)
legend("topright", courses,  fill = color,cex = 0.8, bg = "white")
dev.off() #should be at the end




library(plotrix)
pie3D(x,main = "lpa avg package course-wise", labels = pie_per,col = color)
legend("topright", courses,  fill = color,cex = 0.8, bg = "white")    


library(plotrix)
pie3D(x,main = "lpa avg package course-wise", labels = pie_per,col = color, explode = 0.1)
legend("topright", courses,  fill = color,cex = 0.8, bg = "white")  


#tool and % of meeting that happen using it
val = c(15,20,30)
tools = c("meet","zoom","slack")
color = c("lightblue","lightpink","lightyellow")
pie(per,main = "% of tools used for meetings", labels = tools,col = color)
pie(val)
pie3D(per,main = "% of tools used for meetings", labels = tools,col = color)
dev.off()



BarPlot
#bar
x = c(340,300,200,230)
barplot(x)


#bar
x = c(340,300,200,230)
months = c("jun","jul","aug","sep")
shade = c("lightblue","lightpink","lightyellow","lightgreen")
barplot(x, main = "monthly sales in lakhs", names.arg = months, xlab = "months", ylab = "lakhs", col = shade)
legend("topright",months, fill = shade, cex = 0.8, title = "months")


barplot(x, main = "monthly sales in lakhs", names.arg = months, xlab = "months", ylab = "lakhs", col = shade, horiz = T)
legend("topright",months, fill = shade, cex = 0.8, title = "months")

v1 = c(100,203,400,560)
v2 = c("a","b","c","d")
color = c("lightblue","lightpink","lightyellow","lightgreen")
barplot(v1,main = "Practice",col = color, names.arg = v2)
legend("topright",v2,cex=0.8, fill=color)
dev.off()


Session 7
Data Visualization 
Stacked BarPlot

vm = matrix(v,nrow =3, ncol =5)
mon = c("jan","feb","mar","apr","may")
regions = c("south","north","east")
color = c("lightblue","lightpink","lightyellow")
barplot(vm)
barplot(vm,names.arg = mon)
barplot(vm,names.arg = mon,col =color)
legend("topright",regions,fill=color, cex = 0.8, title= "Sales-RegionWise")
dev.off()




vm = matrix(v,nrow =3, ncol =5)
mon = c("jan","feb","mar","apr","may")
regions = c("south","north","east")
color = c("lightblue","lightpink","lightyellow")
barplot(vm)
barplot(vm,names.arg = mon)
barplot(vm,names.arg = mon,col =color,main = "Sales in crores region wise", beside = T)
legend("topright",regions,fill=color, cex = 0.8, title= "Sales-RegionWise")
dev.off()


Assignment
#5 dept, 3 tools
vm = matrix(v,nrow =3 ,ncol =5)
dept = c("IT","HR","Marketing","Finance", "PR")
tools = c("Teams", "Meet", "Zoom")
barplot(vm)
barplot(vm, main = "Dept wise tools used", col = color, names.arg = dept)
legend("topright",tools, fill = color, cex = 0.8, title = "Tools")
dev.off()


vm = matrix(v,nrow =3 ,ncol =5)
dept = c("IT","HR","Marketing","Finance", "PR")
tools = c("Teams", "Meet", "Zoom")
barplot(vm)
barplot(vm, main = "Dept wise tools used", col = color, names.arg = dept)
barplot(vm, main = "Dept wise tools used", col = color, names.arg = dept, beside = T)
legend("topright",tools, fill = color, cex = 0.8, title = "Tools")
dev.off()



Histogram
#histogram

x = c(26.5,22.4,45.2,23.4,22.3,32.3,34.5,45.3,43.2,34.4,23.3)
hist(x)
hist(x, main = "MCA Students placement package", xlab = "package in lakhs", ylab = "no. of students", col = "cadetblue")



hist_returns = hist(x, main = "MCA Students placement package", xlab = "package in lakhs", ylab = "no. of students", col = "cadetblue")
text(hist_returns$mids,hist_returns$counts,labels= hist_returns$counts, adj = c(0,-0.5))



#frequency graph
hist(x)
hist(x, probability = T)


#showing multiple graph at the same time
par(mfrow = c (1,2)) # where 2 is no of graphs
hist(x,main = "#frequency graph")
hist(x, main="# density graph", probability = T)
par(mfrow = c(1,1))


hist(x, main="# density graph", probability = T)
grid(nx = NULL, ny = NULL, col = "lightblue", lty = 2, lwd = 3)



par(mfrow = c (1,3))
hist(x, main = "default-breaks")
hist(x,breaks = 3, main="3-breaks")
hist(x,breaks = 10, main="10-breaks")
par(mfrow = c(1,1))




hist(x, main = "MCA Students placement package",
     xlab = "package in lakhs", ylab = "no. of students", col = "cadetblue")
hist(y, add = T, col = "lightblue")
legend("topright", c("MCA","MBA"),fill = c("cadetblue", "lightblue"),cex =0.8)
dev.off()


hist(x, main = "MCA Students placement package",
     xlab = "package in lakhs", ylab = "no. of students", col = "cadetblue")
hist(y, add = T, col = "lightblue")
lines(density(x),col= "red",lwd = 3)
lines(density(y),col="blue",lwd =3)
legend("topright", c("MCA","MBA"),fill = c("cadetblue", "lightblue"),cex =0.8)
dev.off()


hist(x, main = "MCA Students placement package",
     xlab = "package in lakhs", ylab = "no. of students", col = "cadetblue", probability = T)
hist(y, add = T, col = "lightblue")
lines(density(x),col= "red",lwd = 3)
lines(density(y),col="blue",lwd =3)
legend("topright", c("MCA","MBA"),fill = c("cadetblue", "lightblue"),cex =0.8)
dev.off()


Assignment
set.seed(123)
x <- rnorm(100, mean = 6, sd = 1.5) 
y <- rnorm(100, mean = 8, sd = 2) 
hist(x, main = "MCA Students placement package",
     xlab = "package in lakhs", ylab = "no. of students", col = "cadetblue", probability = T)
hist(y, add = T, col = "lightblue", probability = T)
lines(density(x),col= "red",lwd = 3)
lines(density(y),col="blue",lwd =3)
legend("topright", c("MCA","MBA"),fill = c("cadetblue", "lightblue"),cex =0.8)
dev.off()



Session 8
Data Visualization
Line Graph / Plots
x = c (10,34,23,45,56,65,34)
plot(x)



x = c (10,34,23,45,56,65,34)
plot(x)
plot(x, type ="l")


x = c (10,34,23,45,56,65,34)
plot(x)
plot(x, type ="l")

y = seq(1,7,by = 1)
plot(x,y,type = "l")


par(mfrow=c(1,4))
plot(x,y, type = "s", main = "type - s")
plot(x,y, type = "l", main = "type - l")
plot(x,y, type = "b", main = "type - b")
plot(x,y, type = "o", main = "type - o")
par(mfrow=c(1,1))
dev.off()


par(mfrow=c(1,4))
plot(x,y, type = "s", main = "type - s", xlab = "X values", ylab = "y values", col= "darkred")
plot(x,y, type = "l", main = "type - l", xlab = "X values", ylab = "y values", col= "navyblue")
plot(x,y, type = "b", main = "type - b", xlab = "X values", ylab = "y values", col= "darkgreen")
plot(x,y, type = "o", main = "type - o", xlab = "X values", ylab = "y values", col= "darkorange")
par(mfrow=c(1,1))


x = c (10,34,23,45,56,65,34)
y = c (18,44,35,53,56,33,24)
z = c (37,24,53,43,55,32,24)
plot(x, type = "o",col="darkred",main = "line graph with multiple lines", xlab = "no of students", ylab = "package in lakhs", lty = 1 , lwd=3)
lines(y ,type = "l", col="darkblue", lty = 3, lwd = 3)
lines(z, type ="l", col="darkgreen", lty =6,lwd = 3)
legend("topleft",lty=c(1,3,6),legend = c("MCA","MBA","MIM"),col = c("darkred","darkblue","darkgreen"), lwd =3, title = "Package Coursewise")




text(6,100,"basline-MIM")
text(13,150,"baseline-MBA")
text(14,170,"baseline-MCA")

Assignment
Create line graphs using any columns from mtcars dataset
data("mtcars")
head(mtcars)
nrow(mtcars)
ncol(mtcars)
x=data("mtcars")
plot(mtcars$mpg,type='l')
plot(mtcars$cyl,type='o')
plot(mtcars$hp,type='s')



Session 9 
Box Plot
x = c(-9,-8,-4,3,5,2,6,8,10,11,12,14,7,9,4,12,19)
boxplot(x)

boxplot(x,horizontal = T, main = "boxplot")



stripchart(x,method = "jitter", add = T, col="darkblue",pch=19)

With outlier
x = c(-9,-8,-4,3,5,2,6,8,10,11,12,14,7,9,4,12,19,49,39,20,34)
boxplot(x)
boxplot(x,horizontal = T, main = "boxplot")
stripchart(x,method = "jitter", add = T, col="darkblue",pch=19)
dev.off()


par(mfrow = c(1,3))
boxplot(x,main="boxplot-vertical")
boxplot(x,notch = T, main="boxplot with notch")
boxplot(x,horizontal = T, main = "boxplot")
stripchart(x,method = "jitter", add = T, col="darkblue",pch=19)

par(mfrow = c(1,1))


head(chickwts)
boxplot(chickwts$weight~chickwts$feed)



head(chickwts)
boxplot(chickwts$weight~chickwts$feed)
stripchart(chickwts$weight~chickwts$feed,data= chickwts, vertical = T, pch = 19,col= 1:length(levels(chickwts$feed)), add = T, method = "jitter")


Assignment - add a legend
head(chickwts)
boxplot(chickwts$weight~chickwts$feed)
stripchart(chickwts$weight~chickwts$feed,data= chickwts, vertical = T, pch = 19,col= 1:length(levels(chickwts$feed)), add = T, method = "jitter")
color = 1:length(levels(chickwts$feed))
legend("topright",title="chickwts", c("casein","horsebean","linseed","meatmeal","soybean","sunflower"), cex = 0.5, fill= color )
dev.off()


y = stack(trees)
y
boxplot(y$values~y$ind)
stripchart(y$values~y$ind, pch=19, vertical =TRUE, col=1:3, 
           add=TRUE, method="jitter")
legend("topright", c("Girth", "Height", "Volume"), fill=c(1, 2, 3),
       title="Tree stats", cex=0.7)


Scatter Plot
#scatter plot
set.seed(123)
x = runif(100)
x
const = rnorm(100,0,0.25)
y = 2 + 3 * x + const
y
plot(x)
plot(x,y)
plot(y~x)
plot(y~x, col = "darkred", pch = 19)



plot(y~x,col = c("black", "pink"), pch = c(11,19)) #diff values/coland pch for x and y

group =  as.factor(ifelse(x<0.5,"group 1","group 2"))
group
plot(y~x, col = 1:length(levels(group)), pch = 1:length(levels(group)))



#fit line/regression line
abline(lm(y~x), col='darkblue',lwd =3, lty =1)



#smooth fit
lines(lowess(y~x), col = "red",lwd = 3, lty = 2)


set.seed(124)
x1 = runif(200)
const = rnorm(200,0,0.01)
y1 = 0.5+2 * x + const

plot(y~x,col="blue", pch =19)
points(y1~x1,col="green",pch = 19)


smoothScatter(y~x, pch =19, col = 'lightblue', cex = 0.8)

smoothScatter(y~x, pch = 19, colramp = colorRampPalette(c("black","cadetblue","lightblue","skyblue")), cex = 0.8)


y = stack(trees)
plot(y$values~y$ind, col = "darkgreen" , pch=19)
smoothScatter(y$values~y$ind, col = "black" , pch=19, cex = 0.8)


head(mtcars)
pairs(~mpg+wt+disp+hp, data = mtcars)


pairs(~mpg+wt+disp+hp, data = mtcars, lower.panel  = NULL)

pairs(~mpg+wt+disp+hp, data = mtcars, upper.panel  = NULL)


x = stack(mtcars)
boxplot(x$values~x$ind)
stripchart(x$values~x$ind, pch=19, vertical =TRUE, col=1:3, add=TRUE, method="jitter")



Session 10
Linear Regression
#simple linear regression
#establishing/ finding the relationship between response and predictor varible
#response/outcome/dependent variable
#predictor/target/independent variable
#linear relationship - either postive or negative

#equation
# y = ax + b
# y - response / dependent var
# b - intercept
# a - slope/coefficients
# x - independent var

yrs = c(1,2,3,4,5,6,7,8,9,10)
sal = c(40000,45000,50000,55000,60000,65000,70000,75000,80000,85000)
data = data.frame(yrs, sal)
#fitting the model
slr_model = lm(sal~yrs,data = data) #lm -> linear model
summary(slr_model)
#1-(2e-16)

#predict the values
predict(slr_model,data.frame(yrs=11))


#y = b + ax
y = 3.500e+04 + (5.000e+03*11)
y



setwd("C:/Users/exam/")
getwd()

df = read.csv("area_prices.csv")
area = df$Area
price = df$Price
data = data.frame(area, price)
slr_model = lm(price~area,data = data)
summary(slr_model)

predict(slr_model,data.frame(area=500))
predict(slr_model,data.frame(area=400))
predict(slr_model,data.frame(area=150))
predict(slr_model,data.frame(area=1500))

summary(slr_model)





mtcars
mlr_model = lm(mpg~cyl+disp+hp+wt,data = mtcars)
summary(mlr_model)
mlr_model_reduced1 = lm(mpg~cyl+disp+hp+wt,data = mtcars)
summary(mlr_model_reduced1)
mlr_model_reduced2 = lm(mpg~hp+wt,data = mtcars)
summary(mlr_model_reduced2)
#annova test for identifying the diff between two models
anova(mlr_model_reduced1, mlr_model_reduced2)

#predict the values
predict(mlr_model_reduced2,data.frame(hp = 111, wt = 2.710))

y = 37.22727 + (-0.03177 * 111) + (-3.87783 * 2.710)
y


df = read.csv("area_prices.csv")
df

area = df$Area
price = df$Price
age = df$Age
bhk = df$BHK
type = df$Type

data = data.frame(area, price,age,bhk, type)

str(data)
as.factor(type)

#to convert categorical
# df$Type = as.factor(df$Type)

mlr_model = lm(price~area+age+bhk+type, data = data)        
summary(mlr_model)
mlr_model_reduced1 = lm(price~area+bhk, data = data)
summary(mlr_model_reduced1)
predict(mlr_model_reduced1, data.frame(area = 700,bhk = 1))
predict(mlr_model, data.frame(area=1500,age = 2, bhk = 3, type = 1))
predict(mlr_model, data.frame(area=1500,age = 4, bhk = 3, type = 1))


Logistic Regression
setwd("C:/Users/exam/34 Falguni Parab/")
getwd()
#logistic Regression
#when your outcome/reponse/dependent var is catogorical
data = read.csv("binary.csv")
data
str(data)
data$admit=as.factor(data$admit)
data$rank=as.factor(data$rank)
str(data)
#Contingency table to check if there is any zero entry in any category

xtabs(~admit+rank,data=data)
#prepare datasets

set.seed(1234)
ind = sample(2,nrow(data),replace=T,prob = c(0.8,0.2))
ind

train = data[ind == 1,]
test = data[ind == 2,]
train

#logistic model fitting
lrm_model = glm(admit~gre+gpa+rank,data = train, family = "binomial")
summary(lrm_model)





pred1 = predict(lrm_model,train,type = "response")
head(pred1)
head(train)
y = -5.009514 + (660* 0.001631) + (3.67 * 1.166408) + (3*-1.125341)
y
exp(y)/(1+exp(y))
p1 = ifelse(pred1>0.5, 1,0)
tab1 = table(Predicted = p1, Actual = train$admit)
tab1
# error your model will make 
1 - sum(diag(tab1)/sum(tab1))



#test
lrm_model = glm(admit~gre+gpa+rank,data = test, family = "binomial")
summary(lrm_model)

pred1 = predict(lrm_model,test,type = "response")
head(pred1)
head(test)

y = -1.586730 + (480* 0.004743) + (3.44 * -0.219328) + (3*-2.263754)
y
exp(y)/(1+exp(y))
p1 = ifelse(pred1>0.5, 1,0)
tab1 = table(Predicted = p1, Actual = test$admit)
tab1
# error your model will make 
1 - sum(diag(tab1)/sum(tab1))

df = read.csv("SocialNetworkADS.csv")
df
#df$Gender = replace(x = df$Gender,list = c("Male", "Female"), values = c(0,1))

df$Gender = ifelse(df$Gender=="Male", 1, 0)
df
str(df)
df$Gender = as.factor(df$Gender)
df$Purchased = as.factor(df$Purchased)
str(df)
set.seed(1234)
ind = sample(2,nrow(df),replace=T,prob = c(0.8,0.2))
ind
train = df[ind == 1,]
test = df[ind == 2,]
train
lrm_model =  glm(Purchased~Gender+Age+EstimatedSalary,data = train, family = "binomial")
summary(lrm_model)

pred1 = predict(lrm_model,train,type = "response")
head(pred1)
head(train)
y = -1.235e+01 + (1* 4.300e-01) + (58000 * 2.255e-01) + (3.631e-05 * 1)
y
exp(y)/(1+exp(y))
p1 = ifelse(pred1>0.5, 1,0)
tab1 = table(Predicted = p1, Actual = train$Purchased)
tab1
# error your model will make 
1 - sum(diag(tab1)/sum(tab1))


Session 11
Web Scraping
#web scrapping with R
#objective: to prepare custom dataset from random webpages
#challenges: permission to web scrape the web scrape the web pages may be restricted

#necessary packages
install.packages("rvest")
install.packages("dplyr")
install.packages("stringr")
install.packages("httr")

#load the packages
library(rvest)
library(dplyr)
library(stringr)
library(httr)

#url
url = "https://editorial.rottentomatoes.com/guide/new-verified-hot-movies/"
page = read_html(url)
page

movie_title = page %>% html_nodes(".article_movie_title a") %>% html_text()
movie_title



year = page %>% html_nodes(".article_movie_title .start-year") %>% html_text()
year


movie_link = page %>% html_nodes(".article_movie_title a") %>% html_attr("href")
movie_link



synopsis = page %>% html_nodes(".synopsis") %>% html_text() %>% str_replace_all("...\n","") %>% str_replace_all("\n","")
synopsis



cast = page %>% html_nodes(".cast") %>% html_text() %>% str_remove_all("Starring:") %>% str_remove_all("\n")
cast




director = page %>% html_nodes(".director") %>% html_text() %>% str_remove_all("Directed By:") %>% str_remove_all("\n")
director



movie_dataset = data.frame(movie_title, movie_link, year, synopsis, cast, director)
movie_dataset

baseurl = "https://books.toscrape.com/catalogue/page-1.html"
page = read_html(baseurl)  
page
book_title = page %>% html_nodes("h3") %>% html_text()
book_title

availability = page %>% html_nodes(".availability") %>% html_text()
availability

link = page %>% html_nodes(".image_container a") %>% html_attr("href")
link




