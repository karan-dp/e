Demo Practical Exam R
programming
Slip no : 1
1. Explain data types in R with suitable
application, code and appropriate comments
1) Numeric
Numeric data types are used for storing numbers, both integers and decimals/doubles.
Code:
# Numeric data type
num1 <- 5 # An integer
num2 <- 5.5 # A decimal/double
paste0("Type of num1: ",typeof(num1))
paste0("Type of num2: ",typeof(num2))
sum_nums <- num1 + num2
print(sum_nums)
paste0("Type of sum: ",typeof(sum_nums))
Output:
2) Integer
Description: Integer data types can be used for integer values. In R Programming Language,
integers are denoted with an "L" at the end of the integer .
Code:
# Integer data type
num1 <- 5L
print(num1)
typeof(num1)
#To display the class type of the object
print(class(num1))
Output:
3) Character/String
Description: Character/String data types store text strings.
Code:
# Character data type
name <- "Jason D’sa" # A string
greeting <- paste("Hello,", name) # Concatenating strings
print(greeting)
typeof(name) Output:
Name: Jason D’sa Class: SY MCA Roll No: 11
4) Logical
Description: Logical data types represent boolean values: TRUE or FALSE.
Code:
# Logical data type
is_valid <- TRUE
print(is_valid)
typeof(is_valid)
# Logical operation
comparison <- (10>=5)
print(comparison)
typeof(comparison)
Output:
5). Vector
Description: Collection of similar types of objects or data components
Code:
#Vector data type
v1 = c(11:28)
v1
v2 = c(11,22,5,13,33,c("jason","abc"))
v2
is.vector(v1)
typeof(v1) #To check the datatype of object
class(v2) #To get the class type of the object
Output:
6) List
Description: Lists can contain elements of different types, including other lists.
Code:
#List data type
rollno = 11:15
name = c("Jason","Tejas","Sharique","Sahil","Prashant")
course = c("MBA","MCA","MIM","MMS","MBA-PT")
rec = list(rollno,names,course)
rec
mat= matrix(seq(10,160,by=10),ncol=4,byrow=TRUE,dimnames =
list(letters[1:4],LETTERS[1:4]))
mat
# Accessing elements in the list
mat[2,]
mat[2,3]
mat[,"A"]
typeof(mat)
7) Data Frame
Description: A data frame is a table-like structure that can hold different data types in each
column.
Code:
#Dataframe Data Type
mca = data.frame(
rollno = 11:15,
names = c("Jason","Tejas","Sharique","Sahil","Prashant"),
city = c("Mumbai","Pune","Mumbai","Pune","Delhi"),
stringsAsFactors = FALSE
)
mca
typeof(mca)
colnames(mca)
rownames(mca)
mca$rollno
Output:
8) Array:
Code:
result = array(
seq(10,480,by=10),
dim = c(4,4,3),
dimnames =
list(c("stud1","stud2","stud3","stud4"),c("java","python","R","android"),c("MBA","MCA","MIM"))
)
result
Output:
2.Demonstrate list data structure of R
programming with suitable examples and all the indexing methods
Code:
rollno = c(11, 15)
name = c("Jason", "Tejas", "Sharique", "Sahil", "Prashant")
course = c("MBA", "MCA", "MIM", "MMS", "MBA-PT")
rec = list(rollno = rollno, names = name, course = course)
print(rec)
# Extracting roll numbers
rec[[1]]
# Extracting names
rec[[2]]
# Extracting a part of a list
rec[1]
# Extracting multiple elements
rec[c(1, 3)]
# Accessing by name
rec[["names"]]
# Accessing using the $ operator
rec$course
Output:
KJ Somaiya Institute of Management, Vidyavihar (East)
Name: Jason D’sa Class: SY MCA Roll No: 11
3.Using (mtcars-mpg,hp,wt) dataset generate informative histogram with suitable legend
Code:
# Load the mtcars dataset
data("mtcars")
pdf("Demo Exam.pdf")
# Set up the plotting area
par(mfrow = c(1, 3)) # 1 row, 3 columns
# Histogram for mpg
hist(mtcars$mpg,
main = "Histogram of MPG",
xlab = "Miles per Gallon",
col = "blue",
border = "black",
xlim = c(10, 35),
breaks = 10)
legend("topright", legend = "MPG", fill = "blue")
# Histogram for hp
hist(mtcars$hp,
main = "Histogram of HP",
xlab = "Horsepower",
col = "green",
border = "black",
xlim = c(50, 300),
breaks = 10)
legend("topright", legend = "HP", fill = "green")
# Histogram for wt
hist(mtcars$wt,
main = "Histogram of Weight",
xlab = "Weight (1000 lbs)",
col = "orange",
border = "black",
xlim = c(1.5, 5.5),
breaks = 10)
legend("topright", legend = "Weight", fill = "orange")
# Reset plotting area
par(mfrow = c(1, 1))
dev.off()
Output:
—----------------------------------------------------------------OR—---------------------------------------------
Question1 - Explain data types in R with suitable
application, code and add appropriate comments
arr1 = array(1:12, dim = c(2, 3, 2))
print(arr1)
arr1[1] # Access the first element
arr1[1, 2, 1] # Access element in the first row, second column, first depth
arr1[arr1 > 5] # Access elements greater than 5
arr1[1, , ] # Access the first row across all columns and depths
m1 = matrix(1:12, nrow = 3, ncol = 4)
print(m1)
m1[1] # Access the first element
m1[2, 3] # Access the element at the second row, third column
m1[1, ] # Access the first row
m1[, 2] # Access the second column
m1[m1 > 5] # Access elements greater than 5
Question2 - Demonstrate list data structure of R programming
with suitable examples and all the indexing methods
l1 = list(a = c(1, 2, 3), b = c(4, 5, 6), c = c(7, 8, 9))
l2 = sapply(l1, mean)
print(l2)
l3 = lapply(l1, function(x) x^2)
print(l3)
df = data.frame(
names = c("A", "B", "C", "D", "E", "F"),
values = c(10, 20, 30, 40, 50, 60)
)
l4 = tapply(df$values, mean)
print(l4)
Question3 - Using trees data structures generate generate
informative line chart (all types of line charts) with suitable
legend
# Load the datasets package which contains the trees data
datasets::trees
# Create a line chart of the volume of trees against the girth
plot(trees$Girth, trees$Volume, type = "l",
main = "Line Chart of Tree Volume vs Girth",
xlab = "Girth", ylab = "Volume",
col = "blue")
legend("topleft", legend = "Tree Volume", col = "blue", lty = 1, cex = 0.8)
# Create a line chart with multiple lines
plot(trees$Girth, trees$Volume, type = "l",
main = "Line Chart of Tree Volume vs Girth",
xlab = "Girth (in)", ylab = "Volume (ft^3)",
col = "blue")
lines(trees$Girth, trees$Height, col = "red")
legend("topright", legend = c("Volume", "Height"), col = c("blue",
"red"), lty = 1, cex = 0.8)
# Set the plot layout to 2 row and 2 columns
par(mfrow = c(2, 3))
# Create the plots
plot(trees$Girth, trees$Height, type = "l", main = "Girth vs Height")
plot(trees$Height,trees$Volume,type="l",main="HeightvsVolume")
plot(trees$Girth,trees$Height,type="l",main="GirthvsHeight")
plot(trees$Height,trees$Volume,type="l",main="HeightvsVolume")
Slip no :2
1 Explain types of functions in R with suitable application, code
and appropriate comments?
1. Predefined Functions (Built-in Functions)
These are functions that come built into R. They are ready to use
# Using the built-in mean() function to calculate the average data
= c(10, 20, 30, 40, 50) average = mean(data) print(average)
sqrt(x)
# Square root sqrt_value
= sqrt(16) sqrt_value
User-defined Functions
These are custom functions created by the user to perform specific tasks.
This function takes a number and returns its square square_function <-
function(x) { return(x^2)
}
# Test the custom function result
<- square_function(5)
print(result)
3. Vectorized Functions
These functions are designed to operate on vectors #
Define a function to double a number
double_function = function(x) {
return(x * 2)
}
# Use sapply to apply the function to a vector
values = c(1, 2, 3, 4, 5) doubled_values
sapply(values, double_function) doubled_values
Output:
2. Demonstrate vector and matrix data structure of R
programming with suitable examples and all the indexing
methods?
#signifiacant data structure of R
#vector is collection of similar types of objects pr data components
is.atomic(a) is.vector(a) is.integer(a) is.double(a)
# 1. Creating a numeric vector numeric_vector
= c(10, 20, 30, 40, 50) numeric_vector
#2. Creating a character vector
v = c("Darshan", "Paras", "Suresh") v
#r will do implicit type casting and make it similar type
vect=c(11,22,5,13,c("Darshan", "Shrutik")) vect
#sequence function
v=seq(1,20) v
#by parameter of seq() decides the step size
v = seq(1,20,by=2) v
#Indexing in vectors #
Retrieve the second element
numeric_vector[2]
# Exclude the first element numeric_vector[-1]
# Retrieve the first and third elements numeric_vector[c(1,
3)]
# Create a named vector
v = c("Darshan" = 22,"Paras" =18, "Suresh" =40)
# Access by name v["Darshan"]
#logical indexing
temp = c(23,5,21,13,34,45,43,23)
month=c("jan","feb","mar","apr","may","june","july")
#setnames will ask 2 parameters nv=c(101,102,103)
setNames(nv,c("R","python","android"))
Multiplication
#append inside the vector
v=c(11,22,13) v[4]=55 v
#modify or update the vector
v=c(11,22,13) v[1]=07 v
Matrix:
#matrix is type of array mat =
matrix(seq(10,120,by=10),nrow = 4,ncol=4) mat
#Creates row wise mat=
matrix(seq(10,160,by=10),ncol=4,byrow=TRUE) mat
#5*5 element mt=
matrix(seq(10,160,by=10),ncol=5,nrow=5,byrow=TRUE) mt
#Indexing in Matrices:
# Access by Row and Column Index:
# Retrieve element from the second row and third column
mat[2, 3]
#Access Entire Row or Column:
# Retrieve the second row mat[2,
]
# Retrieve the third column mat[,
3]
# Retrieve elements from the first two rows and the first two columns
mat[1:2, 1:2]
#dimname uses list
mat= matrix(seq(10,160,by=10),ncol=5,byrow=TRUE,
dimnames = list (letters[1:4],LETTERS[1:5]))
mat
#Appending a vector in the matrix
e = c(170,180,190,200,210)
rbind(mat,e)
f=c(60,110,160,55,110)
mat =cbind(mt,f) mat
#Logical indexing
# Retrieve elements greater than 120
mat[mat > 120]
3 Using (trees-girth,height,volume) dataset generate informative bar graphs
(horizontal, vertical and beside) with suitable legend Code:
# Create a horizontal bar graph for Height
barplot(trees$Height, main = "Tree Heights (Horizontal Bar Plot)", xlab = "Height", ylab
= "Tree Index", col = "lightblue", horiz = TRUE,
legend.text = TRUE)
Output:
Code: barplot(trees$Girth, main = "Tree Girths (Vertical Bar Plot)", xlab = "Tree
Index",ylab =
"Girth", col = "skyblue",
names.arg = 1:nrow(trees),
legend.text = "Girth")
Output:
# Combine Girth and Height into a matrix for side-by-side bars data_matrix
= rbind(trees$Girth, trees$Height)
# Create a beside bar plot for Girth and Height
barplot(data_matrix, beside = TRUE, main = "Tree
Girth and Height (Beside Bar Plot)", col = c("orange",
"purple"), names.arg = 1:nrow(trees), legend.text =
c("Girth", "Height"), xlab = "Tree Index", ylab =
"Values") Output:
Slip No :3
Q1. Explain all conditional statements and loops in R with suitable application, code and
appropriate comments
x = 7
if (x %% 2 == 0) {
print("x is even")
} else {
print("x is odd")
}
numbers = c(22,23,24,25,26,27)
labels = ifelse(numbers %% 2 == 0, "even", "odd")
print(labels)
op = "multiply"
result = switch(op,
"add" = 23+24,
"subtract" = 23-24,
"multiply" = 23*24)
print(result)
for (num in 2:10)
{
is_prime =TRUE
for (i in 2:(num-1)) {
if (num %% i == 0) {
is_prime =FALSE
break
}
}
if (is_prime) {
print(paste(num, "is a prime number"))
}
}
Q2 .Using user inputs generates a simple calculator for at least 5 functionalities based on
user input to perform what?
calculator =function() {
num1 =as.numeric(readline(prompt = "Enter the first number: "))
num2 =as.numeric(readline(prompt = "Enter the second number: "))
cat("Select an operation:\n")
cat("1: Addition (+)\n")
cat("2: Subtraction (-)\n")
cat("3: Multiplication (*)\n")
cat("4: Division (/)\n")
operation =as.numeric(readline(prompt = "Enter the number for the desired operation (1-4): "))
result =switch(operation,
`1` = num1 + num2,
`2` = num1 - num2,
`3` = num1 * num2,
`4` = if (num2 != 0) num1 / num2 else "Division by zero not allowed",
"Invalid operation"
)
cat("The result is: ", result, "\n")
}
calculator()
Q .3 Using HairEyeColor dataset generate informative pie chart with suitable legend
# Load the HairEyeColor dataset
data("HairEyeColor")
# Aggregate the data by hair color
hair_color_data <- margin.table(HairEyeColor, 1) # 1 is for hair color
# Create a pie chart for hair color distribution
pie_colors <- rainbow(length(hair_color_data)) # Define colors for the pie chart
# Create the pie chart
pie(hair_color_data,
main = "Distribution of Hair Colors",
col = pie_colors,
labels = names(hair_color_data))
# Add a legend
legend("bottomleft",
legend = names(hair_color_data), # Use names(hair_color_data) for dynamic labeling
fill = pie_colors,
title = "Hair Color")
Slip No : 4
1. Explain arrays and matrix data structures of R with suitable application and
comments with all indexing techniques
MATRIX m = matrix(seq(6,100,by =6),ncol = 4,byrow
= TRUE)
m
m[3,] #show record row and all column m[2,c(2,4)]
# show second row and 2 and 3 column m[,1] #
show record all rows
ARRAY
a = array(seq(6,360, by=6), dim = c(d1,d2,d3),
dimnames =
list(c("sem1","sem2","sem3","sem4"),c("English","Hindi","Marathi"),c("Adeeba","Shweta"
,"Anjali","Aniket"))
a[,,4]#fourth matrix a[,2,]#second
row of each matrix a[,3,]#Third
column of each matrix
2. Demonstrate sapply, lapppy and tapply functions with suitable data structures
and comments
APPLY
x=seq(10:14) y=seq(15:19)
z=seq(20:24) df =
data.frame(x,y,z) df df =
as.matrix(df)
x=matrix(seq(10:25),nrow=4
) apply(x,1,min)
sapply(y,pow)
lapply(x,pow)
3. Using trees data structures generate generate informative line chart (all types of line
charts) with suitable legend
x = c(20,30,40,50)
pdf("line.pdf") plot(density(x), main="Density Plot", xlab="Values",
ylab="Density",col ="seagreen") lines(density(x),col="skyblue",lwd = 5)
legend("topright",c("Lunch"), fill = c("darkred"), cex = 0.5)
SLIP No : 5
Q1. Explain data frames as a data structure its business applications
and associated inbuilt functions with suitable example
df <- data.frame(
Name = c("Ganesh", "Swamy", "Udit"),
Marks = c(120, 150, 100),
Subject = c("Rprog", "NoSql", "BigData")
)
print (df)
head(df)
tail(df)
df[, "Subject"]
df$Teacher <- c("Sudarshan","Sangeeta","Kirti")
print(df)
df_list <- as.list(df)
print(df_list)
#Added one more row
Q2.Explain all file handling functions to generate necessary
information of the storage and file structure with suitable example
and comments
file1 <- function() {
path1 <- "D:/ganesh/Rprog"
setwd(path1)
repeat {
print("Choose an operation:")
print("1: List all files in the working directory")
print("2: List all directories")
print("3: Check if a file exists")
print("4: Check file size")
print("5: Open file in explorer")
print("6: Open file in RStudio edit mode")
print("7: Check file access mode")
print("8: Open file selection dialog")
print("9: Create a new file")
print("10: Get file info")
print("11: Rename a file")
print("12: Remove a file")
print("13: Read a CSV file")
print("0: Exit")
choice <- as.integer(readline("Enter your choice (0-13): "))
if (choice == 0) {
print("Exiting the program.")
break
}
switch(
choice,
# Option 1: List all files in the working directory
{
print("Files in the working directory:")
print(list.files())
},
# Option 2: List all directories
{
print("Directories in the working directory:")
print(list.dirs())
},
# Option 3: Check if a file exists
{
file_name <- readline("Enter the file name: ")
print(paste("File exists:", file.exists(file_name)))
},
# Option 4: Check file size
{
file_name <- readline("Enter the file name: ")
if (file.exists(file_name)) {
print(paste("File size:", file.size(file_name), "bytes"))
} else {
print("File does not exist.")
}
},
# Option 5: Open file in explorer
{
file_name <- readline("Enter the file name: ")
if (file.exists(file_name)) {
file.show(file_name)
} else {
print("File does not exist.")
}
},
# Option 6: Open file in RStudio edit mode
{
file_name <- readline("Enter the file name: ")
if (file.exists(file_name)) {
file.edit(file_name)
} else {
print("File does not exist.")
}
},
# Option 7: Check file access mode
{
file_name <- readline("Enter the file name: ")
access_mode <- as.integer(readline("Enter access mode (0: read, 2: write, 4: execute): "))
if (file.exists(file_name)) {
print(paste("Access mode available:", file.access(file_name, access_mode) == 0))
} else {
print("File does not exist.")
}
},
# Option 8: Open file selection dialog
{
file.choose()
},
# Option 9: Create a new file
{
new_file <- readline("Enter the new file name: ")
if (!file.exists(new_file)) {
file.create(new_file)
print(paste("Created file:", new_file))
} else {
print("File already exists.")
}
},
# Option 10: Get file info
{
file_name <- readline("Enter the file name: ")
if (file.exists(file_name)) {
print(file.info(file_name))
} else {
print("File does not exist.")
}
},
# Option 11: Rename a file
{
old_name <- readline("Enter the current file name: ")
new_name <- readline("Enter the new file name: ")
if (file.exists(old_name)) {
file.rename(old_name, new_name)
print(paste("Renamed", old_name, "to", new_name))
} else {
print("File does not exist.")
}
},
# Option 12: Remove a file
{
file_name <- readline("Enter the file name to remove: ")
if (file.exists(file_name)) {
file.remove(file_name)
print(paste("Removed file:", file_name))
} else {
print("File does not exist.")
}
},
# Option 13: Read a CSV file
{
file_name <- readline("Enter the CSV file name: ")
if (file.exists(file_name)) {
data <- read.csv(file_name)
print(data)
} else {
print("File does not exist.")
}
},
# Invalid choice
{
print("Invalid choice. Please select a valid option.")
}
)
}
}
file1()
Q3.Using (airquality-ozone,wind,temp) data structure generate
information histogram with suitable labels
data("airquality")
png("q3.png", width = 800, height = 600)
par(mfrow = c(3, 1))
# Ozone Histogram
hist(airquality$Ozone, col="lightblue", main="Ozone Distribution",
xlab="Ozone", ylab="Frequency", border="black")
legend("topright", legend="Ozone", fill="lightblue")
# Wind Histogram
hist(airquality$Wind, col="lightcoral", main="Wind Speed Distribution",
xlab="Wind", ylab="Frequency", border="black")
legend("topright", legend="Wind", fill="lightcoral")
# Temperature Histogram
hist(airquality$Temp, col="lightgreen", main="Temperature Distribution",
xlab="Temperature", ylab="Frequency", border="black")
legend("topright", legend="Temperature", fill="lightgreen")
dev.off()
Slip No: 6
1.Explain data frames as a data structure and use all subset possibilities for searching
the record from dataset
mca <- data.frame(
ID = 1:7,
Name = c("Nidhi", "Bhavesh", "Gauri", "Arpita", "Ritika", "Shwetali", "Nikhil"),
city = c("pune", "mumbai", "delhi", "pune", "chennai", "pune", "kolkata"),
dob = as.Date(c("2002-01-18", "1999-12-10", "2001-11-16", "2002-03-30", "2003-08-25",
"2001-09-10", "1995-12-05"))
)
salary <- c(55000, 65000, 78000, 98000, 87000, 92000, 88000)
mca <- cbind(mca, salary)
mca
# Subset possibilities
# 1. All records from a specified city (e.g., pune)
subset_city <- subset(mca, mca$city %in% c("pune"))
subset_city
# 2. The record with the maximum salary
subset_max_salary <- subset(mca, mca$salary == max(mca$salary))
subset_max_salary
# 3. Records where city is Pune or salary is the maximum
subset_city_or_max_salary <- subset(mca, mca$city %in% c("pune") | mca$salary ==
max(mca$salary))
subset_city_or_max_salary
# 4. Records where DOB is greater than a specified date
subset_dob_after <- subset(mca, mca$dob > as.Date("2001-05-12"))
subset_dob_after
# 5. Retrieve records where DOB is less than a specified date
subset_dob_before <- subset(mca, mca$dob < as.Date("2000-01-12"))
subset_dob_before
2.Demonstrate reading, writing and appending csv,excel and json file for with above
dataframe generated in question 1
library(readxl)
library(writexl)
library(jsonlite)
CSV operations
# 1. Writing the data frame to a CSV file
write.csv(mca, "mca.csv",row.names = FALSE)
read.csv("mca.csv")
# 2. Appending new data to the CSV file
write.table(mca_new, "mca.csv", row.names = FALSE, append = TRUE, col.names = FALSE,
sep = "," )
read.csv("mca.csv")
# 3. Reading data from the CSV file
read.csv("mca.csv")
Excel Operations
# 4. Writing the data frame to an Excel file
write_xlsx(mca, "mcaExcel.xlsx", col_names = TRUE)
read_xlsx("mcaExcel.xlsx")
# 5. Reading data from the Excel file
df <- read_xlsx("mcaExcel.xlsx", sheet = 1,range="A6:C6")
print("Data read from Excel file:")
View(df)
JSON Operations
JSON FILE: mca_new
# 6. Writing the data frame to a JSON file
write_json(mca,"mca1.json")
# 7. Reading data from the JSON file
data=read_json("mca_new.json")
data
data=fromJSON("mca_new.json")
data
Q>3.Using (airquality-ozon,wind,temp) data structure 3 line graphs with suitable labels,
legends and comments
data("airquality")
# Ozone Line Graph
plot(airquality$Ozone, type="l", col="purple", xlab="Days", ylab="Ozone ", main="Ozone
Levels",lty=5)
legend("topright", legend=c("Ozone"), col=c("purple"), lty=5)
# Wind Line Graph
plot(airquality$Wind, type="l", col="red", xlab="Days", ylab="Wind", main="Wind Speed")
legend("topright", legend=c("Wind"), col=c("red"), lty=1)
# Temperature Line Graph
plot(airquality$Temp, type="l", col="grey", xlab="Days", ylab="Temperature",
main="Temperature",lty=4,lwd=2)
legend("topright", legend=c("Temperature"), col=c("grey"), lty=4)
Slip No :7
1 Explain data frames as a data structure and use all subset
possibilities for searching the record from dataset
10
mar
ks
2 Demonstrate reading, writing and appending csv, excel and
json file for with above
dataframe generated in question 1
10
mar
ks
3 Using (air quality-ozone,wind,temp) data structure 3 line
graphs with suitable labels, legends and comments
10
mar
ks
Q>1
Creating Data Frame
CODE :
df = data.frame(
Name = c("Rahul", "Priya", "Amit", "Sneha"),
Game = c("Cricket", "Badminton", "Chess", "Hockey"),
Experience = c(8, 6, 12, 4),
City = c("Pune", "Mumbai", "Pune", "Delhi"),
Salary = c(50000, 60000, 55000, 58000)
)
subset possibilities for searching the record from dataset
CODE : df$Name
CODE :
df[2] # when single index defualt is col index
df[1,] #accessing row of df
df[,3] # accessing Col of df
df[1,2] #second column first value
CODE :
subset(df, df$City %in% c("Pune"))
CODE : subset(df, df$Salary == max(df$Salary))
CODE : subset(df, df$City %in% c("Delhi") | df$Salary == max(df$Salary))
Q>2
CSV
CODE :
Writing to a CSV File:
CODE : write.csv(df, "data.csv", row.names = FALSE)
Reading from a CSV File:
CODE :
df_csv = read.csv("data.csv")
print(df_csv)
Appending to a
CSV File:
CODE :
new_data = data.frame(
Name = c("Karan"),
Game = c("Football"),
Experience = c(7),
City = c("Chennai"),
Salary = c(61000)
)
write.table(new_data, file = "data.csv", sep = ",", col.names = FALSE, append = TRUE,
row.names = FALSE)
Excel
Packages :
install.packages("readxl")
library("readxl")
install.packages("openxlsx")
library(openxlsx)
Writing to an Excel File:
CODE :
write.xlsx(df, "data.xlsx", rowNames = FALSE)
df_excel = read.xlsx("data.xlsx")
print(df_excel)
Appending to an Excel File
CODE :
df_existing <- read.xlsx("data.xlsx")
df_combined <- rbind(df_existing, new_data)
write.xlsx(df_combined, "data.xlsx", rowNames = FALSE) print(df_combined)
Reading from an Excel File:
CODE:
df_excel <- read.xlsx("data.xlsx")
print(df_excel)
JSON
Packages :
install.packages("jsonlite")
library(jsonlite)
Writing to a JSON File:
write_json(df, "data.json")
Reading from a JSON File:
df_json <- fromJSON("data.json")
print(df_json)
Appending to a JSON File:
df_existing_json <- fromJSON("data.json")
df_combined_json <- rbind(df_existing_json, new_data)
write_json(df_combined_json, "data.json")
print(df_combined_json)
Q>3
Loading the Data
data("airquality")
head(airquality)
Creating the Line Graphs
par(mfrow = c(3, 1)) # 3 rows, 1 column
# Plot 1: Ozone Levels
pdf("ozone.pdf")
plot(airquality$Ozone, type = "l", col = "red", lwd = 2, xlab = "Day of Observation", ylab = "Ozone
(ppb)", main = "Ozone Levels Over Time")
legend("topright", legend = "Ozone", col = "red", lty = 1, lwd = 2) dev.off()
# Plot 2: Wind Speed
pdf("wind.pdf")
plot(airquality$Wind, type = "l", col = "yellow", lwd = 2,
xlab = "Day of Observation", ylab = "Wind (mph)",
main = "Wind Speed Over Time")
legend("topright", legend = "Wind", col = "yellow", lty = 1, lwd = 2)
dev.off()
# Plot 3: Temperature Levels
pdf("temp.pdf")
plot(airquality$Temp, type = "l", col = "purple", lwd = 2,
xlab = "Day of Observation", ylab = "Temperature (°F)",
main = "Temperature Over Time")
legend("topright", legend = "Temperature", col = "purple", lty = 1, lwd = 2)
dev.off()
Slip No :8
1 Demonstrate how to generate vectors with
different types methods along with all indexing
methods
10
marks
2 Generate IPL dataframe for 10 teams and write it
into a json file. Perform all the subset operations on
IPL dataset including subset for date etc.
10
marks
3 Generate dataset for height and weight of
mca,mba and msc students (60 students hint-use
randomizer) and generate group bar chart with all
possible visualization with labels, legends and
comments
10
marks
Q1.Demonstrate how to generate vectors with different types methods along with all
indexing methods
a.Using c()
# Numeric vector
numeric_vector <- c(1, 2, 3, 4, 5)
numeric_vector
# Character vector
char_vector <- c("A", "B", "C", "D")
char_vector
#Logical vector
logical_vector <- c(TRUE, FALSE, TRUE)
logical_vector
b.Using seq()
v=seq(1,20)
v
c.By ()parameter
v=seq(1,20,by=3)
v
d. Creating a vector by repeating a value
vr=rep(10, times = 10)
vr
e.Creating an Empty Vector
ve=c()
Ve
f.Random Vector Generation
rv=rnorm(6,mean = 5, sd = 2)
rv
g.Run if
rf=runif(7,min =10,max =15)
rf
h.Indexing Method
v[2]
v[-2] #show all except second
v[2:4] #range as index
temp = c(3,55,32,43,3,5,43)
month = c("jan","feb","mar","apr","may","june","july")
month[temp<20]
Q2.Generate IPL dataframe for 10 teams and write it into a json file. Perform all the
subset operations on IPL dataset including subset for date etc.
Code:
install.packages("jsonlite")
library(jsonlite)
df = data.frame(Team = c("CSK","MI","KKR","SH","RCB","KP","DD","GT","LS","RR"),
Wins = c(13:6, 10, replace = TRUE),
Losses = c(2:9, 10, replace = TRUE),
Dates = sample(seq(as.Date('2024-04-01'), as.Date('2024-06-01'), by="day"),
10, replace = TRUE)
)
df
j=write_json(df, "ipl_data.json")
df_json<-fromJSON("ipl_data.json")
print(df_json)
#Subset by Team
csk = subset(df, Team == "CSK")
csk
#Subset by Wins More Than 10
wins_more = subset(df, Wins > 10)
wins_more
#Subset by Losses Less Than 8
loss_less = subset(df, Losses < 8)
loss_less
#Subset by Dates after 2024-05-01
dates_after = subset(df, Dates > as.Date('2024-05-01'))
dates_after
Q3.Generate dataset for height and weight of mca,mba and msc students (60 students
hint-use randomizer) and generate group bar chart with all possible visualization with
labels, legends and comments.
Code:
mca = data.frame(
rollno = 1:60,
height = sample(120:220, 60),
weight = sample(50:110, 60)
)
mca
mba = data.frame(
rollno = 1:60,
height = sample(100:220, 60),
weight = sample(50:110, 73)
)
mba
msc = data.frame(
rollno = 1:60,
height = sample(100:220, 200),
weight = sample(50:110, 64)
)
msc
height_data <- msc$height
weight_data <- msc$weight
rollno <- msc$rollno
# Create a barplot for heights
barplot(height_data,
names.arg = rollno,
col = "skyblue",
main = "Bar Plot of Heights in MSC",
xlab = "Roll Numbers",
ylab = "Height (cm)",
las = 2)
# Create a barplot for weights
barplot(weight_data,
names.arg = rollno,
col = "lightcoral",
main = "Bar Plot of Weights in MSC",
xlab = "Roll Numbers",
ylab = "Weight (kg)",
las = 2)
par(mfrow=c(1,2))
plot(height_data,type="l",main = "Line",col='orange',xlab= 'value')
plot(weight_data,type="s",main="type s",col='yellow',xlab= 'value')
hist(height_data,main = "MSC Height Data",col='brown')
#Scatter Plot
plot(height_data, weight_data,
main ="Height/Weight Scatter Plot",
xlab ="Height MSC",
ylab =" Weight MSC ", pch = 19)
#pie Chart
pie(height_data,weight_data,col = rainbow(length(height_data)))
#Horizontal barplot
barplot(weight_data,
names.arg = rollno,
col = "lightblue",
main = "Bar Plot of Weights in MSC",
xlab = "Roll Numbers",
ylab = "Weight (kg)",
horiz=TRUE,
las = 2)
#Multi value historgram
hist(height_data,main="MSC students Height Data",xlab="Height in cm ",col =
"aquamarine",probability = TRUE)
hist(weight_data,add=TRUE,col="lightblue",probability = TRUE)
lines(density(height_data),col="red",lwd=3)
lines(density(weight_data),col="blue",lwd=3)
Slip no : 9
1. Demonstrate arrays data structure with 4 matrices of
5 x 5 size each, for scores of 5 players in 5 different
sports rows(player-1,player-2,player-3,player-4,player-5) and
columns(sports-1,sports-2,sports-3,sports-4,sports-5)
Code:
sports=array(seq(1,5,by=1),dim =
c(5,5),dimnames=list(c("player-1","player-2","player-3","player-4","player-5"),
c("sports-1","sports-2","sports-3","sports-4","sports-5")))
sports
Output:
2.Explain all the types of functions in R with suitable
example and comments
Simple Function
wish = function(){
print("good morning guys")
}
wish()
Parameterized Function
wish = function(name , institute){
paste("good morning " , name , "you are from: " , institute)
}
wish("pratham" , "KJSomaiya")
Changing the order of parameters
wish(institute="Birla" , name = "Pratham")
Default parameterized function
def= function(name , institute , class="MCA")
{
paste("You are from class " , class)
}
def("Pratham" , "KJSim")
Return statement in function
ret = function(a,b)
{
return(a+b)
}
ret(4,5)
Function returning multiple values
area = function(l,h)
{
area = (l+h)
perimeter = (l*h)
circum =(2*3.14*(l*h))
result = list(area=area , perimeter=perimeter , circum=circum)
return(result)
}
result = area(5,7)
paste(result["area"] , result["perimeter"] , result["circum"])
Inline Function
pow = function(x,n) x ^n
pow(2,2)
Apply function
x = seq(1:5)
Y = seq(6:10)
z = seq(11:15)
df = data.frame(x,Y,z)
df
apply(df , 1 , sum)
apply(df , MARGIN = 1 , sum)
3.Generate dataset for height and weight of mca,mba
and msc students (60 students hint-use randomizer)
and generate line chart with all possible visualization
with labels, legends and comments
student_data =data.frame(
mca_height = round(runif(n_students, min = 150, max = 190), 1),
mca_weight = round(runif(n_students, min = 50, max = 100), 1),
mba_height = round(runif(n_students, min = 150, max = 190), 1),
mba_weight = round(runif(n_students, min = 50, max = 100), 1),
msc_height = round(runif(n_students, min = 150, max = 190), 1),
msc_weight = round(runif(n_students, min = 50, max = 100), 1)
)
plot(student_data[,1],xlab ="height",type="l",col="red")
lines(student_data[,3],type="o",col="blue")
lines(student_data[,5],type="s",col="green")
legend("topright",title = "Height",legend = c("MCA","MBA","MSC"),fill
=c("red","blue","green"))
plot(student_data[,2],xlab ="weight",type="l",col="red")
lines(student_data[,4],type="o",col="blue")
lines(student_data[,6],type="s",col="green")
legend("topright",title = "weight",legend = c("MCA","MBA","MSC"),fill
=c("red","blue","green"))
Slip No:10
Q.1.) Demonstrate list data structure with for scores of 5 players in 5 different sports
rows(player-1,player-2,player-3,player-4,player-5) and
columns(sports-1,sports-2,sports-3,sports-4,sports-5) Convert it into dataframe
Perform - append row, append column with functions and indexing both
Code:
scores = list(
Player1 = c(85, 90, 78, 92, 88),
Player2 = c(80, 85, 95, 70, 90),
Player3 = c(88, 92, 85, 91, 86),
Player4 = c(75, 80, 82, 84, 79),
Player5 = c(90, 88, 92, 87, 91)
)
sports = c("Sport1", "Sport2", "Sport3", "Sport4", "Sport5")
df = as.data.frame(do.call(rbind, scores))
rownames(df) = names(scores)
colnames(df) = sports
print("Initial Data Frame:")
print(df)
#append row
new_player_scores = c(82, 84, 88, 90, 85)
df = rbind(df, NewPlayer = new_player_scores)
print("Data Frame after appending player:")
print(df)
#append column
new_sport_scores = c(87, 92, 85, 80, 90, 88)
df = cbind(df, Sport6 = new_sport_scores)
print("Data Frame after appending sport:")
print(df)
#column indexing
player_3_scores = df["Player3", ]
print("Scores of Player 3:")
print(player_3_scores)
#row indexing
sport_2_scores = df[, "Sport2"]
print("Scores for Sport 2:")
print(sport_2_scores)
Output:
Q.2.) Generate dynamic calculator with user inputs for performing min 7 functionalities
including continuing and exiting from the calculator
Code:
calculator = function() {
repeat {
cat("Select an operation:\n")
cat("1. Addition\n")
cat("2. Subtraction\n")
cat("3. Multiplication\n")
cat("4. Division\n")
cat("5.Exit\n")
choice = as.integer(readline(prompt = "Enter your choice (1-5): "))
if (choice == 5) {
cat("Exiting the calculator")
break
}
num1 = as.numeric(readline(prompt = "Enter first number: "))
num2 = as.numeric(readline(prompt = "Enter second number: "))
result = NULL
switch(choice,
`1` = { result = num1 + num2; operation = "Addition" },
`2` = { result = num1 - num2; operation = "Subtraction" },
`3` = { result = num1 * num2; operation = "Multiplication" },
`4` = {
if (num2 == 0) {
result = "Error: Division by zero"
} else {
result = num1 / num2
}
operation = "Division"
},
{
cat("Invalid choice. Please select a valid operation.\n")
}
)
if (!is.null(result)) {
cat(paste(operation, "result:", result, "\n"))
}
}
}
calculator()
Output:
Q.3.) Generate nice bar charts - horizontal, vertical and beside with ChickWeight dataset
with suitable label and legends
Code:
df=ChickWeight
df
mean_weights_time = aggregate(ChickWeight[, "weight"] ~ ChickWeight[, "Time"], FUN = mean)
colnames(mean_weights_time) = c("Time", "Mean_Weight")
pdf("Plots.pdf")
# Vertical Bar Plot
barplot(mean_weights_time[, "Mean_Weight"], names.arg = mean_weights_time[, "Time"],
xlab = "Time",
ylab = "Mean Weight",
main = "Mean Chick Weight Over Time (Vertical)", col = "lightblue")
# Horizontal Bar Plot
barplot(mean_weights_time[, "Mean_Weight"], names.arg = mean_weights_time[, "Time"],
xlab = "Mean Weight",
ylab = "Time",
main = "Mean Chick Weight Over Time (Horizontal)", horiz = TRUE,
col = "lightgreen")
dev.off()
Output:
