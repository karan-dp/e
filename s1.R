setwd("C:/Users/kpanchal/Desktop/Sem//3/R")

getwd()
#list all the files
list.files()

#list directories 
list.dirs()
#. is for hidden directories

#to check if file exists
file.exists("abc.txt")
file.exists("car-dataset.xlsx")

#check size of file
file.size("car-dataset.xlsx")
file.size("abc.txt")


#opens the file in appropriate explorer
file.show("test.txt")

#opens the file in R studio in edit mode
file.edit("test.txt")

#opens the file in appropriate explorer
file.show("car-dataset.xlsx")

#opens the file in R studio in edit mode
file.edit("test.txt")

