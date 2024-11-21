Slip Number 1 

Q.1) Explain data types in R with suitable application, code and appropriate comments.

Code:
  # Numeric example
  num <- 42.5  # A decimal value
integer_num <- 10  # An integer value

# Performing a calculation
sum <- num + integer_num
print(sum)  # Output: 52.5

# Character example
name <- "John Doe"  # A string
greeting <- paste("Hello,", name)  # Concatenating strings
print(greeting)  # Output: "Hello, John Doe"

# Integer example
count <- as.integer(100)  # Explicitly defining an integer
print(count)  # Output: 100

# Complex example
complex_num <- 3 + 4i  # Complex number with real and imaginary parts
print(complex_num)  # Output: 3+4i

# Data frame with multiple data types
data <- data.frame(
  Name = c("Alice", "Bob"),
  Age = c(25, 30),  # Numeric
  Gender = factor(c("Female", "Male"))  # Factor
)

print(data)


Q.2) Demonstrate list data structure of R programming with suitable examples and all the indexing methods.

Code:
  # Example of a list with different data types
  my_list <- list(
    name = "Alice",          # Character
    age = 25,                # Numeric
    scores = c(90, 85, 88),  # Numeric Vector
    passed = TRUE            # Logical
  )

print(my_list)

#Using the $ Operator
# Accessing by name
print(my_list$name)  # Output: "Alice"
print(my_list$scores)  # Output: 90 85 88

#Access elements by position or name.
# Accessing by position
print(my_list[[1]])  # Output: "Alice"

# Accessing by name
print(my_list[["age"]])  # Output: 25

# Extracting a sublist
sublist <- my_list[1]
print(sublist)  # Output: A list containing "Alice"

# To access the element, use [[ ]] or $
print(sublist[[1]])  # Output: "Alice"

# Modify an element
my_list$age <- 26
print(my_list$age)  # Output: 26

# Add a new element
my_list$city <- "New York"
print(my_list$city)  # Output: "New York"

# Combine two lists
list1 <- list(a = 1, b = 2)
list2 <- list(c = 3, d = 4)
combined_list <- c(list1, list2)

print(combined_list)

# Using a loop
for (item in my_list) {
  print(item)
}

# Using lapply to apply a function to each element
result <- lapply(my_list, class)
print(result)  # Output: Classes of each element

Q.3) Using (mtcars-mpg,hp,wt) dataset generate informative histogram with suitable legend
Code:
  # Load the mtcars dataset
  data(mtcars)

# Set up the plotting area to show multiple histograms in one figure
par(mfrow = c(1, 3))  # 1 row, 3 columns for side-by-side histograms

# Histogram for Miles Per Gallon (mpg)
hist(mtcars$mpg, 
     main = "Histogram of MPG", 
     xlab = "Miles Per Gallon", 
     col = "blue", 
     border = "black")
legend("topright", legend = "MPG", fill = "blue", cex = 0.8)

# Histogram for Horsepower (hp)
hist(mtcars$hp, 
     main = "Histogram of HP", 
     xlab = "Horsepower", 
     col = "red", 
     border = "black")
legend("topright", legend = "HP", fill = "red", cex = 0.8)

# Histogram for Weight (wt)
hist(mtcars$wt, 
     main = "Histogram of Weight", 
     xlab = "Weight (1000 lbs)", 
     col = "green", 
     border = "black")
legend("topright", legend = "Weight", fill = "green", cex = 0.8)

# Reset the plotting area
par(mfrow = c(1, 1))


Slip Number 2

Q. 1) Explain types of functions in R with suitable application, code and appropriate comments
Code:
  # Function in R
  wish = function(){
    print("Good Morning Guys")
  }
wish()

# Function
wish = function(name,institute){
  paste("Good Morning Guys",name,"You are from: ",institute)
}
wish("MCAFY","KJSIM")

# Function
wish = function(name,institute){
  paste("Good Morning Guys",name,"You are from: ",institute)
}
wish(institute="Somaiya University",name="MCASY2023-25")

#lazy evaluation of a function
info = function(name,institute,city){
  paste("Good Morning Guys",name,"You are from: ",institute)
}
info(institute="Somaiya University",name="MCASY2

# Default Parameterized function
info = function(name="John"){
  paste("My name is:",name)
}
info()

# function returning value
sum = function(a,b){
  return (a+b)
}
sum(5,6)

 Function returning multiple values
area= function(l,h){
  area = (l+h)
  parameter = (l*h)
  circumference = (2*3.14+(l*h))
  result = list(area=area,parameter=parameter,circumference=circumference)
  return(result)
}

result = area(5,7)
paste("area:",result["area"],"parameter:",
      result["parameter"],"circumference:",result["circumference"])


# Inline function
pow = function(x,n) x^n
pow(2,3)


Q.2)  Demonstrate vector and matrix data structure of R programming with suitable examples and all the indexing methods
Code:
#append values in vectors
v= c(11,22,33,44)
v[5]=55
v


#modify the vectors
v=c(11,22,33,44)
v[2]=55
v


#remove the vector
v=0
v
v=c(11,22,33,44)
rm(v)
v

# range 
v= 15:23
v

v[-2]
v

v= v[-2]
v

#accessing vectors
v= c(15,16,17,18,19,20,21,22,23)
v

v[2] #atomic value

v[-2] #show all except second value

v[2:4] #range as indexing

v[c(2,4,6)] #vector defined with c() as indexing

v[seq(2,4)] #vector defined with seq() as indexing

v[c(TRUE,FALSE)] #comparing values 

v[c(TRUE,FALSE,TRUE,TRUE)] 

#logical indexing
temp= c(23,5,21,14,43,34)
month= c("jan","feb","march","april","may","jun")
month[temp<20]

month[temp<20 | temp==43]

month[temp<20 | temp>43]




Q.3) Using (trees-girth,height,volume) dataset generate informative bar graphs (horizontal, vertical and beside) with suitable legend

Code:

# Load the trees dataset (this is built-in in R)
data(trees)

# 1. Vertical Bar Chart for Girth, Height, and Volume
barplot(height = as.matrix(trees[, c("Girth", "Height", "Volume")]), 
        beside = FALSE,  # Stacked bars
        col = c("lightblue", "pink", "lightgreen"),  # Different colors for each column
        main = "Vertical Bar Chart for Trees Dataset",  # Title of the chart
        xlab = "Trees",  # Label for the x-axis
        ylab = "Measurement Values",  # Label for the y-axis
        legend = c("Girth", "Height", "Volume"),  # Legend
        args.legend = list(x = "topright"))  # Position the legend at the top-right

# 2. Horizontal Bar Chart for Girth, Height, and Volume
barplot(height = as.matrix(trees[, c("Girth", "Height", "Volume")]), 
        beside = FALSE,  # Stacked bars
        col = c("lightblue", "pink", "lightgreen"),  # Different colors for each column
        main = "Horizontal Bar Chart for Trees Dataset",  # Title of the chart
        xlab = "Measurement Values",  # Label for the x-axis
        ylab = "Trees",  # Label for the y-axis
        horiz = TRUE,  # Make the bars horizontal
        legend = c("Girth", "Height", "Volume"),  # Legend
        args.legend = list(x = "topright"))  # Position the legend at the top-right

# 3. Beside Bar Chart for Girth, Height, and Volume
barplot(height = as.matrix(trees[, c("Girth", "Height", "Volume")]), 
        beside = TRUE,  # Grouped bars (side-by-side)
        col = c("lightblue", "pink", "lightgreen"),  # Different colors for each column
        main = "Beside Bar Chart for Trees Dataset",  # Title of the chart
        xlab = "Trees",  # Label for the x-axis
        ylab = "Measurement Values",  # Label for the y-axis
        legend = c("Girth", "Height", "Volume"),  # Legend
        args.legend = list(x = "topright"))  # Position the legend at the top-right



Slip Number 3 

Q.1) Explain all conditional statements and loops in R with suitable application, code and appropriate comments
Code:

# Conditional Statements 
age <- as.integer(readline("Enter your age: "))

if (age < 18) {
  print("You are a minor.")
} else if (age < 65) {
  print("You are an adult.")
} else {
  print("You are a senior.")
}

Code:

#  For Loop: print Letters 
X = LETTERS
for(i in LETTERS){
  print(i)
}

Code:

# while loop : Print numbers from 1 to 5 using while loop
count <- 1
while (count <= 5) {
  print(count)
  count <- count + 1
}


Q.2) Using user inputs generates a simple calculator for at least 5 functionalities based on user input to perform what?
Code:

num= TRUE
while (num) {
  a= as.integer(readline("Enter number 1:"))
  b= as.integer(readline("Enter number 2:"))
  choose= readline(prompt = "Choose operation to perform or type 'exit': ")
  
  if (choose == "+") {
    print(paste("Addition: ", a + b))
  } 
  else if (choose == "-") {
    print(paste("Substraction: ", a - b))
  } 
  else if (choose == "*") {
    print(paste("Multiplication: ", a * b))
  } 
  else if (choose == "/") {
    print(paste("Division: ", a / b))
  } 
  else if (choose == "exit") {
    print("Exiting the program.")
    num= FALSE
  } 
  else {
    print("Invalid input. Please try again.")
  }
}


Q.3) Using HairEyeColor dataset generate informative pie chart with suitable legend

Code:

# Step 1: Summarize the data by hair color
hair_colors <- rowSums(HairEyeColor)

# Step 2: Create the pie chart
pie(hair_colors, 
    main = "Hair Color Distribution", 
    col = c("black", "brown", "red", "yellow"), 
    labels = c("Black", "Brown", "Red", "Blonde"))

# Step 3: Add a legend
legend("topright", 
       legend = c("Black", "Brown", "Red", "Blonde"), 
       fill = c("black", "brown", "red", "yellow"), 
       cex = 0.8)
Slip Number 4


Q.1) Explain arrays and matrix data structures of R with suitable application and comments with all indexing techniques

# Array
demo = array(
  seq(10,480,by=10),
  dim= c(4,4,3),
)
print(demo)


Indexing Techniques

# Single Indexing
print(demo[1, 1, 1])


# Multiple Indexing
print(demo[c(1, 2), c(1, 2), c(1, 2)])


# Range Indexing
print(demo[1:2])


# Dimension Indexing
print(demo[, 1, 1])


# Logical Indexing
print(demo[demo<400])


# Matrix
mt <- matrix(seq(10, 160, by = 10), nrow = 4, ncol = 4)
print(mt) 


# Print a specific element
print(mt[4, 5])


# Print a row
print(mt[1, ]) 


# Print a column
print(mt[, 3]) 


# Print a range of rows
print(mt[1:4, ]) 


# Print a range of rows in reverse order
print(mt[4:1, ]) 


# Print specific columns of a row
print(mt[2, c(2, 3)]) 


Q.2) Demonstrate sapply, lapppy and tapply functions with suitable data structures and comments

#List of numbers
numbers <- list(1:5, 6:10, 11:15)
numbers


#sapply(List of sum)
s_apply <- sapply(numbers,sum)
s_apply


#lappy(Vector of sum)
l_apply <- lapply(numbers, sum)
l_apply


#tapply
exam_scores <- c(85, 90, 78, 92, 88, 76, 95, 89)
subjects <- factor(c("Math", "Math", "Science", "Science", "Math", "Science", "Math", "Science"))
mean <- tapply(exam_scores, subjects, mean)
print(mean)


Q.3) Using trees data structures generate informative line chart (all types of line charts) with suitable legend

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
legend("topright", legend = c("Volume", "Height"), col = c("blue", "red"), lty = 1, cex = 0.8)

# Create the multiple plots
par(mfrow = c(1, 4))
plot(trees$Girth, trees$Height, type = "l", col= 'pink', main = "Girth vs Height")
plot(trees$Height, trees$Volume, type = "l", col= 'cadetblue', main = "Height vs Volume")
plot(trees$Girth, trees$Height, type = "l", col= 'maroon', main = "Girth vs Height")
plot(trees$Height, trees$Volume, type = "l", col= 'grey', main = "Height vs Volume")
Slip Number 5

Q.1) Explain data frames as a data structure its business applications and associated inbuilt functions with suitable example
Business Applications of Data Frames
Healthcare and Research
Sales and Marketing:
Financial Analysis:
Customer Data Management:

Code:
# Create a data frame
data <- data.frame(
  Name = c("John", "Alice", "Bob", "Emma"),
  Age = c(28, 24, 30, 22),
  Salary = c(45000, 52000, 48000, 46000)
)

# Display the data frame
print(data)

str() - Structure of the Data Frame
str(data)

summary() - Summary Statistics
summary(data)

head() and tail() - Viewing Top and Bottom Rows
head(data)  # First 6 rows (default)
tail(data)  # Last 6 rows (default)

names() or colnames() - Column Names
names(data)

dim() - Dimensions of the Data Frame
dim(data)

subset() - Subsetting the Data Frame
# Extract rows where Age > 25
subset_data <- subset(data, Age > 25)
print(subset_data)

apply() - Applying Functions Over Data Frame
# Apply mean function to each column
apply(data[ , 2:3], 2, mean)

merge() - Merging Data Frames
data2 <- data.frame(
  Name = c("John", "Alice", "Emma"),
  Department = c("Sales", "HR", "Marketing")
)

# Merge two data frames by the Name column
merged_data <- merge(data, data2, by = "Name")
print(merged_data)




Q.2) Explain all file handling functions to generate necessary information of the storage and file structure with suitable example and comments
Code:
# set working directory
setwd("C:\\Users\\kpanchal\\Desktop\\R_EXAM")

# get working directory
getwd()

# List all the files
list.files()

# Show list of directory
list.dirs()

# Check if file exist or not if exist returns -TRUE
file.exists("mca.txt")

# Check file size
file.size("mca.txt")

# opens the file in appropriate explorer
file.show("mca.txt")

# opens the file in r studio in edit mode
file.edit("mca.txt")

#check if the mode is available
file.access("mca.txt",6)

#append a file
file.append("mca.txt","mca.txt")

#opens the dialog box to choose file
file.choose()

#Create a file
file.create("test.txt")

#Copy contents from one file to another
file.copy("mca.txt","test.txt",overwrite = TRUE)

#get the file info
file.info("mca.txt")

#gives the permission details for user group
file.mode("mca.txt")

#rename the file
file.rename("test1.txt","demo.txt")

#remove files
file.remove("test.doc")



Q.3) Using (airquality-ozon,wind,temp) data structure generate information histogram with suitable labels and legend

Slip Number 6

Q.1) Explain data frames as a data structure and use all subset possibilities for searching the record from dataset

df <- data.frame(
  # Name =c("A", "B","C", "D","E", "F"),
  Age = c(22,22,22,22,22,22),
  Gender = c("M", "M", "F", "M", "F", "M")
)
print(df)

df[1, ]
# First row
df[, 2]
# Second column
df[1:2, 1:2] 

# Select specific columns by name
df["Name"]
df[["Name"]]
df$Name

# Select rows where Age > 30
subset(df, Age > 30)
# Select specific rows based on a condition
df[df$Age > 30,]

Q.2) Demonstrate reading, writing and appending csv, excel and json file for with above dataframe
generated in question 1



Q.3) Using (airquality-ozon,wind,temp) data structure 3 line graphs with suitable labels, legends and comments

Slip Number 7

Q.1 Explain data frames as a data structure and use all subset possibilities for searching the record from dataset

data <- data.frame(
  Name = c("Pranjal","Karan","Falguni","Vijay", "Pratik", "Nilesh"),
  Age = c(28, 24, 35, 32, 34,21)
)


print(data)
print(data[data$Age > 30, ])
print(data[1, ])
print(data$Name)
print(data[2, "Name"])
print(subset(data, Age < 30 & Name == "Pranjal"))



Q.2 Demonstrate reading, writing and appending csv, excel and json file for with above dataframe generated in question 1
write.csv(data, "data1.csv", row.names = FALSE)
data_csv <- read.csv("data1.csv")
print(data_csv)
library(xlsx)
write.xlsx(data, "data1.xlsx", row.names = FALSE)
data_excel <- read.xlsx("data1.xlsx", 1)
print(data_excel)
library(jsonlite)
write_json(data, "data1.json")
data_json <- read_json("data1.json")
print(data_json)
data_append <- data.frame(
  Name = c("New"),
  Age = c(40)
)
write.table(data_append, "data1.csv", row.names = FALSE, col.names = FALSE, append = TRUE, sep = ",")
data_new <- read.csv("data1.csv")
print(data_new)

Q.3 Using (airquality-ozon,wind,temp) data structure 3 line graphs with suitable labels, legends and comments
airquality
data("airquality")
par(mfrow=c(1,3))
plot(airquality$Ozone, type="l", col="darkblue", 
     main="Ozone Levels", xlab="Day", ylab="Ozone (ppb)")
plot(airquality$Wind, type="l", col="darkred", 
     main="Wind Speed", xlab="Day", ylab="Wind Speed (mph)")
plot(airquality$Temp, type="l", col="darkgreen", 
     main="Temperature", xlab="Day", ylab="Temperature (F)")
legend("topright", legend=c("Ozone", "Wind", "Temp"), 
       col=c("darkblue", "darkred", "darkgreen"),lty = 1, cex=0.8)
par(mfrow=c(1,1))
dev.off()



plot(airquality$Ozone, type="l", col="darkblue", 
     main="Ozone Levels", xlab="Day", ylab="Ozone")
lines(airquality$Wind, type = "l", col="darkred")
lines(airquality$Temp, type = "l", col="darkgreen")
legend("topright", legend=c("Ozone", "Wind", "Temp"), 
       col=c("darkblue", "darkred", "darkgreen"),lty = 1, cex=0.8)




Slip Number 8

Q.1) Demonstrate how to generate vectors with different types methods along with all indexing methods

Code:
# 1. Create vectors using different methods
num_vector <- c(1, 2, 3, 4, 5)
char_vector <- c("Apple", "Banana", "Cherry")
log_vector <- c(TRUE, FALSE, TRUE, FALSE)

# Create a sequence vector
seq_vector <- seq(1, 10, by = 2)

# Repeat vector elements
rep_vector <- rep(5, 3)
rep_elements_vector <- rep(c(1, 2, 3), each = 3)

# Create an empty vector
empty_vector <- vector("numeric", length = 5)

# 2. Indexing Methods
# Basic Indexing
second_element <- num_vector[2]
subset_elements <- num_vector[c(1, 3)]

# Negative Indexing
exclude_third <- num_vector[-3]

# Logical Indexing
log_index <- c(TRUE, FALSE, TRUE, TRUE, FALSE)
filtered_vector <- num_vector[log_index]

# Named Indexing
names(num_vector) <- c("A", "B", "C", "D", "E")
named_element <- num_vector["C"]

# Using `which()`
indices <- which(num_vector > 2)

# 3. Modifying Vectors
num_vector[2] <- 10
num_vector[c(1, 3)] <- c(100, 200)
num_vector["D"] <- 50

# Output results for reference
print(num_vector)
print(second_element)
print(subset_elements)
print(exclude_third)
print(filtered_vector)
print(named_element)
print(indices)


Q.2) Generate IPL dataframe for 10 teams and write it into a json file. Perform all the subset operations on IPL dataset including subset for date etc.

Code:
# Load necessary library
install.packages("jsonlite")

library(jsonlite)  # For working with JSON files

# 1. Generate the IPL dataset for 10 teams
set.seed(123)  # Set seed for reproducibility

# Sample data for IPL teams
teams <- c("Mumbai Indians", "Chennai Super Kings", "Delhi Capitals", "Kolkata Knight Riders", 
           "Royal Challengers Bangalore", "Punjab Kings", "Rajasthan Royals", "Sunrisers Hyderabad", 
           "Kochi Tuskers Kerala", "Deccan Chargers")

# Random data for matches
dates <- seq(as.Date("2024-03-01"), by = "days", length.out = 10)  # 10 match dates
match_number <- 1:10
home_team <- sample(teams, 10, replace = TRUE)
away_team <- sample(teams, 10, replace = TRUE)
result <- sample(c("Home Win", "Away Win", "Draw"), 10, replace = TRUE)

# Create the dataframe
ipl_data <- data.frame(
  Match_No = match_number,
  Date = dates,
  Home_Team = home_team,
  Away_Team = away_team,
  Result = result
)

# Print the first few rows of the IPL dataset
head(ipl_data)

# 2. Write the IPL dataset to a JSON file
write_json(ipl_data, path = "ipl_data.json", pretty = TRUE)

# 3. Subset operations on the IPL dataset
# Subset by date - matches played after a certain date
subset_by_date <- subset(ipl_data, Date > as.Date("2024-03-05"))
print("Subset by date (after 2024-03-05):")
print(subset_by_date)

# Subset for a specific team (e.g., Mumbai Indians as Home Team)
subset_mumbai_home <- subset(ipl_data, Home_Team == "Mumbai Indians")
print("Subset where Mumbai Indians is the Home Team:")
print(subset_mumbai_home)

# Subset for a specific result (e.g., Away Team won)
subset_away_win <- subset(ipl_data, Result == "Away Win")
print("Subset where Away Team won:")
print(subset_away_win)

# Subset for specific columns (e.g., Date and Result columns)
subset_columns <- subset(ipl_data, select = c(Date, Result))
print("Subset for Date and Result columns:")
print(subset_columns)


Q.3) Generate dataset for height and weith of mca,mba and msc students (60 students hint-use randomizer) and generate group bar chart with all possible visualization with labels, legends and comments

Code:

# Load necessary library for plotting
install.packages("tidyr")
library(ggplot2)  # For creating visualizations

# 1. Generate the dataset for Height and Weight of MCA, MBA, and MSC students
set.seed(123)  # Set seed for reproducibility

# Courses
courses <- c("MCA", "MBA", "MSC")

# Random height (in cm) and weight (in kg) for 60 students
height <- round(runif(60, 150, 190), 1)  # Random heights between 150 and 190 cm
weight <- round(runif(60, 45, 90), 1)    # Random weights between 45 and 90 kg

# Course factor for each student (repeating the course names 20 times for 60 students)
course <- rep(courses, each = 20)

# Create the data frame
student_data <- data.frame(
  Student_ID = 1:60,
  Course = course,
  Height = height,
  Weight = weight
)

# View the first few rows of the dataset
head(student_data)

# 2. Create a Grouped Bar Chart for Height and Weight by Course

# Convert data to long format for grouped bar chart
library(tidyr)  # For reshaping data
long_data <- student_data %>%
  pivot_longer(cols = c("Height", "Weight"), names_to = "Measurement", values_to = "Value")

# Grouped bar chart
bar_plot <- ggplot(long_data, aes(x = Course, y = Value, fill = Measurement, group = Measurement)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +  # Create bars for each measurement
  labs(
    title = "Height and Weight of Students by Course",  # Title of the chart
    x = "Course",  # Label for the x-axis
    y = "Value (cm / kg)",  # Label for the y-axis
    fill = "Measurement"  # Legend title for the fill color
  ) +
  theme_minimal() +  # Minimal theme to clean the background
  theme(legend.position = "top")  # Position the legend at the top

# Print the grouped bar chart
print(bar_plot)
Slip Number 9

Q.1) Demonstrate arrays data structure with 4 matrices of 5 x 5 size each, for scores of 5 players in 5 different sports rows(player-1,player-2,player-3,player-4,player-5) and columns(sports-1,sports-2,sports-3,sports-4,sports-5)

Code:
# Create a 3D array of scores for 5 players in 5 sports across 4 matrices
scores <- array(c(
  # Matrix 1: Scores for Players 1 to 5 in Sports 1 to 5
  10, 20, 30, 40, 50,
  15, 25, 35, 45, 55,
  12, 22, 32, 42, 52,
  18, 28, 38, 48, 58,
  14, 24, 34, 44, 54,
  
  # Matrix 2: Scores for Players 1 to 5 in Sports 1 to 5
  11, 21, 31, 41, 51,
  16, 26, 36, 46, 56,
  13, 23, 33, 43, 53,
  19, 29, 39, 49, 59,
  17, 27, 37, 47, 57,
  
  # Matrix 3: Scores for Players 1 to 5 in Sports 1 to 5
  12, 22, 32, 42, 52,
  17, 27, 37, 47, 57,
  14, 24, 34, 44, 54,
  20, 30, 40, 50, 60,
  18, 28, 38, 48, 58,
  
  # Matrix 4: Scores for Players 1 to 5 in Sports 1 to 5
  13, 23, 33, 43, 53,
  18, 28, 38, 48, 58,
  15, 25, 35, 45, 55,
  21, 31, 41, 51, 61,
  19, 29, 39, 49, 59
), dim = c(5, 5, 4))

# Display the entire 3D array
print(scores)

# Accessing specific elements from the array
# Get the score of Player 3 in Sport 2 for the 1st matrix
score <- scores[3, 2, 1]
print(score)

# Get the entire scores for Player 2 across all 4 matrices (in all sports)
player2_scores <- scores[2, , ]
print(player2_scores)

# Get all the scores for Sport 4 across all players and matrices
sport4_scores <- scores[, 4, ]
print(sport4_scores)


Q.2) Explain all the types of functions in R with suitable example and comments

Code:
# 1. Built-in Functions
# Using a built-in mathematical function to calculate square root
x <- 16
sqrt_x <- sqrt(x)  # Finds the square root of 16
print(sqrt_x)  # Output: 4

# Using a built-in statistical function to calculate the mean
data <- c(10, 20, 30, 40, 50)
mean_data <- mean(data)  # Finds the average of the data
print(mean_data)  # Output: 30

# 2. User-Defined Functions
# Defining a function to calculate the square of a number
square_number <- function(x) {
  result <- x^2  # Square of x
  return(result)  # Return the result
}

# Using the user-defined function
result <- square_number(5)
print(result)  # Output: 25

# 3. Anonymous Functions
# Using an anonymous function inside the 'sapply' function
data <- c(1, 2, 3, 4, 5)
result <- sapply(data, function(x) x^3)  # Cube each element
print(result)  # Output: 1 8 27 64 125

# 4. Functions with Default Arguments
# Function with default arguments
greet <- function(name = "Guest", age = 30) {
  message <- paste("Hello", name, "you are", age, "years old!")
  return(message)
}

# Calling function with and without arguments
greeting1 <- greet("Alice", 25)  # Provides both arguments
greeting2 <- greet()  # Uses default values
print(greeting1)  # Output: "Hello Alice you are 25 years old!"
print(greeting2)  # Output: "Hello Guest you are 30 years old!"

# 5. Recursive Functions
# Recursive function to calculate factorial
factorial_recursive <- function(n) {
  if (n == 1) {
    return(1)  # Base case
  } else {
    return(n * factorial_recursive(n - 1))  # Recursive call
  }
}

# Calling the recursive function
result <- factorial_recursive(5)  # 5! = 5 * 4 * 3 * 2 * 1 = 120
print(result)  # Output: 120

# 6. Higher-Order Functions
# Function that accepts a function as an argument and applies it
apply_function <- function(f, x) {
  return(f(x))  # Apply the function f to x
}

# Using the higher-order function with a built-in function 'sqrt'
result <- apply_function(sqrt, 16)  # Apply 'sqrt' function to 16
print(result)  # Output: 4

# 7. Vectorized Functions
# Vectorized operation to calculate the square of each element
data <- c(1, 2, 3, 4, 5)
squared_data <- data^2  # Square each element in the vector
print(squared_data)  # Output: 1 4 9 16 25

# 8. Functions Returning Multiple Values
# Function that returns multiple values using a list
multi_return <- function(x, y) {
  result_sum <- x + y
  result_diff <- x - y
  return(list(sum = result_sum, difference = result_diff))
}

# Calling the function and extracting values
results <- multi_return(10, 5)
print(results$sum)        # Output: 15
print(results$difference) # Output: 5


Q.3) Generate dataset for height and weight of mca,mba and msc students (60 students hint-use randomizer) and generate line chart with all possible visualization with labels, legends and comments

Code:

# 1. Generate the dataset for Height and Weight of MCA, MBA, and MSC students
set.seed(123)  # Set seed for reproducibility

# Random height (in cm) and weight (in kg) for 60 students
height <- round(runif(60, 150, 190), 1)  # Random heights between 150 and 190 cm
weight <- round(runif(60, 45, 90), 1)    # Random weights between 45 and 90 kg

# Create the data frame with Student ID and the generated data
student_data <- data.frame(Student_ID = 1:60, Height = height, Weight = weight)

# View the first few rows of the dataset
head(student_data)

# 2. Basic Line Chart for Height using ggplot2 (without geom_line() and geom_point())
library(ggplot2)

# Simple Line Chart for Height with minimal background
height_plot <- ggplot(student_data, aes(x = Student_ID, y = Height)) +
  geom_line(colour = "blue") +  # Only line, no points
  labs(
    title = "Height of Students",
    x = "Student ID",
    y = "Height (cm)"
  ) +
  theme_minimal()  # Removes the grey background

# Print the height line chart
print(height_plot)

# 3. Basic Line Chart for Weight using ggplot2 (without geom_line() and geom_point())
weight_plot <- ggplot(student_data, aes(x = Student_ID, y = Weight)) +
  geom_line(colour = "cadetblue") +  # Only line, no points
  labs(
    title = "Weight of Students",
    x = "Student ID",
    y = "Weight (kg)"
  ) +
  theme_minimal()  # Removes the grey background

# Print the weight line chart
print(weight_plot)



Slip Number 10

Q.1) Demonstrate list data structure with for scores of 5 players in 5 different sports rows(player-1,player-2,player-3,player-4,player-5) and columns(sports-1,sports-2,sports-3,sports-4,sports-5). Convert it into dataframe Perform -  append row, append column with functions and indexing both

Code:
Step 1: Create List with Scores of 5 Players in 5 Different Sports
# Create a list with scores of 5 players in 5 sports
scores_list <- list(
  Player1 = c(10, 20, 30, 40, 50),  # Scores of player 1
  Player2 = c(15, 25, 35, 45, 55),  # Scores of player 2
  Player3 = c(12, 22, 32, 42, 52),  # Scores of player 3
  Player4 = c(18, 28, 38, 48, 58),  # Scores of player 4
  Player5 = c(14, 24, 34, 44, 54)   # Scores of player 5
)

# Display the list
scores_list

Step 2: Convert List to DataFrame
# Convert the list to a dataframe
scores_df <- as.data.frame(scores_list)

# Display the dataframe
scores_df

Step 3: Append Row to the DataFrame
# Create a new row with player 6's scores
new_row <- c(16, 26, 36, 46, 56)

# Append the new row using rbind()
scores_df <- rbind(scores_df, new_row)

# Display the updated dataframe
scores_df

Step 4: Append Column to the DataFrame
# Create a new column (for example, scores in a new sport)
new_column <- c(60, 65, 70, 75, 80, 85)

# Append the new column to the dataframe
scores_df$Sport6 <- new_column

# Display the updated dataframe
scores_df

Step 5: Indexing the DataFrame
# Accessing specific rows and columns using indexing
# Example: Accessing the score of Player 3 in Sport 2
score_player3_sport2 <- scores_df[3, 2]

# Display the score
score_player3_sport2

Q.2) Generate dynamic calculator with user inputs for performing min 7 functionalities including continuing and exiting from the calculator

Code:
# Function to display the calculator menu and perform operations
calculator <- function() {
  
  repeat {
    # Display options to the user
    cat("\n--- R Calculator ---\n")
    cat("1. Add\n")
    cat("2. Subtract\n")
    cat("3. Multiply\n")
    cat("4. Divide\n")
    cat("5. Modulo (Remainder)\n")
    cat("6. Exponentiation\n")
    cat("7. Square Root\n")
    cat("8. Exit\n")
    
    # Ask the user to select an option
    choice <- as.integer(readline(prompt = "Enter your choice (1-8): "))
    
    # Check if the choice is within the valid range
    if (choice < 1 || choice > 8) {
      cat("Invalid choice. Please select a valid option.\n")
      next
    }
    
    # If user chooses to exit, break the loop
    if (choice == 8) {
      cat("Exiting the calculator. Goodbye!\n")
      break
    }
    
    # Get user input for numbers
    num1 <- as.numeric(readline(prompt = "Enter the first number: "))
    num2 <- as.numeric(readline(prompt = "Enter the second number: "))
    
    # Perform the corresponding operation based on the user's choice
    result <- NULL
    switch(choice,
           `1` = result <- num1 + num2,           # Addition
           `2` = result <- num1 - num2,           # Subtraction
           `3` = result <- num1 * num2,           # Multiplication
           `4` = if (num2 != 0) result <- num1 / num2 else cat("Error: Division by zero!\n"),  # Division
           `5` = result <- num1 %% num2,          # Modulo
           `6` = result <- num1^num2,             # Exponentiation
           `7` = if (num1 >= 0) result <- sqrt(num1) else cat("Error: Cannot calculate square root of negative number!\n")  # Square Root
    )
    
    # Print the result if a valid operation was performed
    if (!is.null(result)) {
      cat("Result: ", result, "\n")
    }
    
    # Ask the user if they want to continue
    continue <- readline(prompt = "Do you want to continue? (y/n): ")
    if (tolower(continue) != 'y') {
      cat("Exiting the calculator. Goodbye!\n")
      break
    }
  }
}

# Call the calculator function
calculator()


Q.3) Generate nice bar charts - horizontal, vertical and beside with ChickWeight dataset with suitable label and legends

Code:
Step 1: Load the Dataset and ggplot2 Package
install.packages("ggplot2")

# Load the necessary library
library(ggplot2)

# Load the ChickWeight dataset
data("ChickWeight")

Step 2: Vertical Bar Chart (Bar Plot)
# Vertical bar chart (with diet and average weight)
ggplot(ChickWeight, aes(x = Diet, y = weight, fill = Diet)) +
  stat_summary(fun = "mean", geom = "bar", show.legend = TRUE) +
  labs(title = "Average Chick Weight by Diet (Vertical Bar Chart)",
       x = "Diet",
       y = "Average Weight (grams)") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3")
Step 3: Horizontal Bar Chart
# Horizontal bar chart
ggplot(ChickWeight, aes(x = Diet, y = weight, fill = Diet)) +
  stat_summary(fun = "mean", geom = "bar", show.legend = TRUE) +
  coord_flip() +  # Flips the chart to horizontal
  labs(title = "Average Chick Weight by Diet (Horizontal Bar Chart)",
       x = "Average Weight (grams)",
       y = "Diet") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3")
Step 4: Bar Chart with Bars Beside Each Other
# Bar chart with bars beside each other
ggplot(ChickWeight, aes(x = Time, y = weight, fill = Diet)) +
  geom_bar(stat = "identity", position = "dodge") +  # Position bars beside each other
  labs(title = "Chick Weight Over Time by Diet (Beside Bar Chart)",
       x = "Time (Days)",
       y = "Weight (grams)") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3")


Slip Number 11

Q.1) Demonstrate all the data structures with different ways of creating them, different indexing methods and other supporting functions

Code:

# 1.Vector(Homogeneous, 1-D)
# Creating vectors
vec1 <- c(1, 2, 3, 4, 5)       # Numeric
vec2 <- c("A", "B", "C")       # Character
vec3 <- seq(1, 10, by = 2)     # Sequence

# Indexing
vec1[1]                        # First element
vec1[c(2, 4)]                  # Multiple elements
vec1[vec1 > 3]                 # Logical indexing

# Functions
length(vec1)                   # Length
sum(vec1)                      # Sum
mean(vec1)                     # Mean

# 2.List(Heterogeneous, 1-D)
# Creating lists
lst <- list(name = "John", age = 25, scores = c(95, 87, 92))

# Indexing
lst[[1]]                       # First element
lst[["name"]]                  # By name
lst$scores                     # Using $

# Functions
names(lst)                     # Get names of list elements
length(lst)                    # Length of the list

# 3.Matrix(Homogeneous, 2-D)
# Creating matrices
mat <- matrix(1:9, nrow = 3, byrow = TRUE)  # By rows

# Indexing
mat[1, 2]                      # Row 1, Column 2
mat[, 2]                       # Entire second column
mat[2, ]                       # Entire second row

# Functions
dim(mat)                       # Dimensions of the matrix
t(mat)                         # Transpose
rowSums(mat)                   # Sum of rows

# 4.Array(Homogeneous, Multi-Dimensional)
# Creating arrays
arr <- array(1:12, dim = c(2, 3, 2))        # 3D array

# Indexing
arr[1, 2, 1]                   # Element at [1, 2, 1]
arr[, , 1]                     # All elements from the 1st matrix

# Functions
dim(arr)                       # Dimensions
apply(arr, c(1, 2), sum)       # Apply function along margins

# 5.Data Frame(Heterogeneous, 2-D)
# Creating a data frame
df <- data.frame(
  ID = 1:3,
  Name = c("Alice", "Bob", "Charlie"),
  Scores = c(88, 92, 95)
)

# Indexing
df[1, 2]                       # First row, second column
df$Name                        # Access 'Name' column
df[df$Scores > 90, ]           # Subset rows

# Functions
str(df)                        # Structure
summary(df)                    # Summary statistics

# 6.Factor(Categorical Data)
# Creating a factor
fct <- factor(c("Low", "Medium", "High", "Medium", "Low"))

# Indexing
fct[1]                         # First element
levels(fct)                    # Levels of the factor

# Functions
table(fct)                     # Frequency table
is.factor(fct)                 # Check if it's a factor



Q.2) Demonstrate all the types of graphs we learned till now with suitable legends , labels and comments using any inbuilt or custom dataset

Code:

# Load a built-in dataset
data(mtcars)  # Motor Trend Car Road Tests dataset

# Custom dataset for demonstration
custom_data <- data.frame(
  Category = c("A", "B", "C", "D"),
  Values = c(10, 15, 7, 12)
)

# 1.Scatter Plot**
plot(mtcars$wt, mtcars$mpg,
     main = "Scatter Plot of Weight vs. MPG",
     xlab = "Weight (1000 lbs)",
     ylab = "Miles Per Gallon",
     pch = 19, col = "blue")
legend("topright", legend = "Data Points", col = "blue", pch = 19)

# 2.Line Plot**
plot(custom_data$Values, type = "o",
     main = "Line Plot of Custom Data",
     xlab = "Index",
     ylab = "Values",
     col = "red", lty = 1, pch = 16)
legend("topright", legend = "Values", col = "red", lty = 1, pch = 16)

# 3.Bar Plot**
barplot(custom_data$Values, names.arg = custom_data$Category,
        main = "Bar Plot of Categories",
        xlab = "Categories",
        ylab = "Values",
        col = rainbow(4))
legend("topright", legend = custom_data$Category, fill = rainbow(4))

# 4.Histogram**
hist(mtcars$mpg,
     main = "Histogram of Miles Per Gallon",
     xlab = "MPG",
     col = "green", border = "black")
legend("topright", legend = "Frequency", fill = "green")

# 5.Box Plot**
boxplot(mpg ~ cyl, data = mtcars,
        main = "Box Plot of MPG by Cylinder Count",
        xlab = "Number of Cylinders",
        ylab = "Miles Per Gallon",
        col = c("lightblue", "pink", "lightgreen"))
legend("topright", legend = c("4 Cyl", "6 Cyl", "8 Cyl"),
       fill = c("lightblue", "pink", "lightgreen"))

# 6.Pie Chart**
pie(custom_data$Values, labels = custom_data$Category,
    main = "Pie Chart of Custom Data",
    col = rainbow(4))
legend("topright", legend = custom_data$Category, fill = rainbow(4))

# 7.Boxplot with points**
boxplot(mtcars$hp, main = "Horsepower Distribution", xlab = "Horsepower")
points(jitter(rep(1, nrow(mtcars))), mtcars$hp, col = "red", pch = 16)
legend("topright", legend = "Data Points", col = "red", pch = 16)





# Simple linear regression 
 years_experience = c(1,2,3,4,5,6,7,8,9,10)
 salary = c(40000,45000,50000,55000,60000,65000,70000,75000,80000,85000)
 data = data.frame(years_experience,salary)
 data

 #Fitting the model
 slr_model = lm(salary~years_experience,data=data)
 #Summary of the model
 summary(slr_model)

code:
 #Predciting the value
 predict(slr_model,data.frame(years_experience=11))

#y = b +ax
 y = 3.500e+04 + (5.000e+03*11) #11 years of experince
 y


 #Multiple Linear Regression
 #equation
 y = a0x0 + a1x1 + a2x2 + ......anxn + b
 mtcars
 mlr_model = lm(mpg~cyl+disp+hp+wt,data=mtcars)

 mlr_model_reduced1 = lm(mpg~disp+hp+wt,data=mtcars) #After removing cyl(cylinder as it was
 not conctributing)
 summary(mlr_model_reduced1)

 mlr_model_reduced2 = lm(mpg~hp+wt,data=mtcars) #After removing disp(disp as it was not
 conctributing)
 summary(mlr_model_reduced2)

 #Anova test for identifying the difference between two models
 anova(mlr_model_reduced1,mlr_model_reduced2)

 #Predict the values
 predict(mlr_model_reduced2,data.frame(hp=111,wt=2.710))

 #Cross Check the values using equation y = a0x0 + a1x1 + a2x2 + ......anxn + b
 y = 37.22727 + (-0.03177*111) + (-3.87783*2.710)
 y 





