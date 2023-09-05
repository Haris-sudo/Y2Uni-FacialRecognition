DROP DATABASE IF EXISTS StudentRecog;

CREATE DATABASE StudentRecog;

CREATE TABLE StudentRecog.School (
	School_ID int(11) NOT NULL,
	School_Name varchar(30),
	Address_Line_1 varchar(30),
	Address_Line_2 varchar(30),
	Postcode varchar(8),
	Phone_num int(15),
	PRIMARY KEY(School_ID)
);

CREATE TABLE StudentRecog.Courses (
	Course_ID int(11) NOT NULL AUTO_INCREMENT,
	School_ID int(11),
	Course_Name varchar(30),
	Course_Level int(1),
	Start_Date date,
	End_Date date,
	PRIMARY KEY(Course_ID),
	FOREIGN KEY(School_ID) REFERENCES School(School_ID)
)AUTO_INCREMENT=1;

CREATE TABLE StudentRecog.Students (
	Student_ID int(11) NOT NULL AUTO_INCREMENT,
	First_Name varchar(30),
	Last_Name varchar(30),
	DOB date,
	Gender varchar(15) NOT NULL,
	Address_Line_1 varchar(30),
	Address_Line_2 varchar(30),
	Postcode varchar(8),
	Phone_Num int(15),
	Email_Address varchar(40) DEFAULT NULL,
	Mask_Exemption BOOLEAN NOT NULL,
	Course_ID int,
	PRIMARY KEY(Student_ID),
	FOREIGN KEY(Course_ID) REFERENCES Courses(Course_ID)
)AUTO_INCREMENT=88;

CREATE TABLE StudentRecog.Modules (
	Module_ID int(11) NOT NULL AUTO_INCREMENT,
	Course_ID int(11),
	Module_Name varchar(30),
	Start_Date date,
	End_Date date,
	PRIMARY KEY(Module_ID),
	FOREIGN KEY(Course_ID) REFERENCES Courses(Course_ID)
)AUTO_INCREMENT=1;

CREATE TABLE StudentRecog.Lecturers (
	Lecturer_ID int(11) NOT NULL AUTO_INCREMENT,
	Lecturer_First_Name varchar(30),
	Lecturer_Last_Name varchar(30),
	DOB date,
	Gender varchar(15) NOT NULL,
	Address_Line_1 varchar(30),
	Address_Line_2 varchar(30),
	Postcode varchar(8),
	Phone_Num int(15),
	Email_Address varchar(40) DEFAULT NULL,
	username varchar(45),
    password varchar(90),
	PRIMARY KEY(Lecturer_ID)
)AUTO_INCREMENT=1;

CREATE TABLE StudentRecog.Lectures (
	Lecture_ID int(11) NOT NULL AUTO_INCREMENT,
	Module_ID int(11),
	Lecturer_ID int,
	Start_DateTime datetime(6),
	End_DateTime datetime(6),
	PRIMARY KEY(Lecture_ID),
	FOREIGN KEY(Module_ID) REFERENCES Modules(Module_ID),
	FOREIGN KEY(Lecturer_ID) REFERENCES Lecturers(Lecturer_ID)	
)AUTO_INCREMENT=1;

CREATE TABLE StudentRecog.Attendance_Register (
	Register_ID int (11) NOT NULL AUTO_INCREMENT,
	Student_ID int(11),
	Lecture_ID int(11),
	Present boolean,
	PRIMARY KEY(Register_ID),
	FOREIGN KEY(Student_ID) REFERENCES Students(Student_ID),
	FOREIGN KEY(Lecture_ID) REFERENCES Lectures(Lecture_ID)
)AUTO_INCREMENT=1;


INSERT INTO StudentRecog.School (School_ID, School_Name, Address_Line_1, Address_Line_2, Postcode, Phone_Num)
values
('1', 'The University of Lincoln', 'Brayford Way', 'Brayford Pool', 'LN61 7TS', '0774522628');

INSERT INTO StudentRecog.Courses (School_ID, Course_Name, Course_Level, Start_Date, End_Date)
values
('1', 'Computer Science', '1', '2020/08/15', '2023/05/15'),
('1', 'Computer Science', '2', '2020/08/15', '2023/05/15'),
('1', 'Computer Science', '3', '2020/08/15', '2023/05/15');

INSERT INTO StudentRecog.Students (First_Name, Last_Name, DOB, Gender, Address_Line_1, Address_Line_2, Postcode, Phone_Num, Email_Address, Mask_Exemption, Course_ID)
values
('James','Smith','1996/11/24','Male','Address line 1','Address line 2', 'PH51 6EL', '0773917498', 'Jamessmith@students.lincoln.ac.uk', false, '1'),
('Sarah','Wood','1995/12/19','Female','Address line 1','Address line 2', 'GK24 6DF', '0778267492', 'Sarahwood@students.lincoln.ac.uk', true, '1'),
('Harrison','Goddard','19900/09/13','Male','Address line 1','Address line 2', 'DL37 2DQ', '0779283027', 'Harrisongoddard@students.lincoln.ac.uk', false, '1'),
('Jimmy','Smith','2001/11/12','Male','Address line 1','Address line 2', 'SK21 1KS', '0777839478', 'Jimmysmith@students.lincoln.ac.uk', false, '1'),
('Max', 'West', '1999/04/21', 'Male', 'Address line 1', 'Address line 2', 'LD32 2DA', '0772917846', 'Maxwest@students.lincoln.ac.uk', false, '2'),
('Sally', 'Weaton', '2000/11/18', 'Female', 'Address line 1', 'Address line 2', 'DL29 2DK', '0778176392', 'Sallyweaton@students.lincoln.ac.uk', false, '2'),
('Fenton', 'Alan', '1995/04/12', 'Male', 'Address line 1', 'Address line 2', 'DK18 1SK', '0777163615', 'Fentonalan@students.lincoln.ac.uk', true, '2'),
('Louise', 'Western', '1992/06/19', 'Female', 'Address line 1', 'Address line 2', 'SO29 2SL', '0773912738', 'Louisewestern@students.lincoln.ac.uk', false, '2'),
('Claire', 'Dunn', '2000/07/18', 'Female', 'Address line 1', 'Address line 2', 'SL12 3SD', '0779327103', 'Clairedunn@students.lincoln.ac.uk', true, '3'),
('Charlie', 'Maxwell', '2002/11/16', 'Male', 'Address line 1', 'Address line 2', 'SJ13 1SL', '0772839173', 'Charliemaxwell@students.lincoln.ac.uk', true, '3'),
('Alex', 'Westbrook', '1993/11/21', 'Male', 'Address line 1', 'Address line 2', 'DL23 5SD', '0771937162', 'Alexwestbrook@students.lincoln.ac.uk', true, '3'),
('Matthew', 'Broughton', '2002/07/12', 'Male', 'Address line 1', 'Address line 2', 'AL21 1SL', '0771836182', 'Matthewbroughton@students.lincoln.ac.uk', true, '3');

INSERT INTO StudentRecog.Modules (Course_ID, Module_Name, Start_Date, End_Date)
values
('1', 'Programming Fundamentals', '2020/08/15', '2020/12/03'),
('1', 'Algorithms and Complexity', '2020/08/15', '2020/12/13'),
('1', 'Computer Architectures', '2020/12/03', '2021/03/18'),
('1', 'Maths for Computing', '2020/12/13', '2021/05/28'),
('1', 'Problem Solving', '2020/08/01', '2021/02/14'),
('2', 'Advanced Programming', '2020/08/15', '2020/12/03'),
('2', 'Artifical Intelligence', '2020/08/15', '2020/12/13'),
('2', 'Network Fundamentals', '2020/12/03', '2021/05/26'),
('2', 'Scalable Databases', '2020/12/13', '2021/05/28'),
('2', 'User Experience', '2020/12/01', '2021/05/24'),
('3', 'Big Data', '2020/08/15', '2020/12/03'),
('3', 'Machine Learning', '2020/08/15', '2020/12/13'),
('3', 'Parallel Programming', '2020/12/03', '2021/05/25'),
('3', 'Project', '2020/12/13', '2021/05/21'),
('3', 'Option', '2020/08/01', '2021/02/14');

INSERT INTO StudentRecog.Lecturers (Lecturer_First_Name, Lecturer_Last_Name, DOB, Gender, Address_Line_1, Address_Line_2, Postcode, Phone_Num, Email_Address, username, password)
values
('John', 'Smith', '1976/04/23', 'Male', 'Address line 1', 'Address line 2', 'LS21 2DQ', '0771936789', 'Johnsmith@staff.lincoln.ac.uk', 'jsmith', 'password1'),
('Sarah', 'Johnson', '1986/11/18', 'Female', 'Address line 1', 'Address line 2', 'LN12 2FD', '0778273017', 'Sarahjohnson@staff.lincoln.ac.uk', 'sjohnson', 'password2'),
('Jim', 'Eastman', '1960/04/13', 'Male', 'Address line 1', 'Address line 2', 'AJ27 2AW', '0771937461', 'Jimeastman@staff.lincoln.ac.uk', 'jeastman', 'password3'),
('Adam', 'Sharp', '1979/08/18', 'Male', 'Address line 1', 'Address line 2', 'SK23 1FJ', '0778163846', 'Adamsharp@staff.lincoln.ac.uk', 'asharp', 'password4'),
('Laura', 'Green', '1976/11/20', 'Female', 'Address line 1', 'Address line 2', 'KD13 5DF', '0772936718', 'Lauragreen@staff.lincoln.ac.uk', 'lgreen', 'password5');

INSERT INTO StudentRecog.Lectures (Module_ID, Lecturer_ID, Start_DateTime, End_DateTime)
values
('4', '1', '2021-05-28 08:00:00', '2021-05-28 10:00:00'),
('13', '4', '2021-05-27 08:00:00', '2021-05-27 10:00:00'),
('10', '5', '2021-05-26 08:00:00', '2021-05-26 10:00:00');

INSERT INTO StudentRecog.Attendance_Register ()
values
();