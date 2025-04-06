import os
import pickle
import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image
from streamlit_option_menu import option_menu
import numpy as np
Attrition_model = pickle.load(open("E:\Employee Attrition Analysis and Prediction\Employee_Attrition.sav", "rb"))
Promotion_model = pickle.load(open("E:\Employee Attrition Analysis and Prediction\Employee_Promotion.sav", "rb"))

with st.sidebar:
    selected = option_menu('Employee Attrition Analysis and Prediction',
                           
                           ['Employee Attrition Prediction',
                            'Employee Promotion Prediction'],
                            menu_icon='employee-fill',
                            icons= ['employee', 'job', 'performace', 'Promotion'],
                            default_index = 0)
    

#Employee Attrition Prediction
if selected == 'Employee Attrition Prediction':
    
    st.title(":red[Employee Attrition Prediction using ML]")
    # image = Image.open('Kidney.jpg')
    # st.image(image, caption='Kidney disease')
    name = st.text_input("Name:")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        Age = st.number_input('Age')

    with col2:
        DailyRate = st.number_input('DailyRate')

    with col3:
        Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        if Department == "Sales":
            Department = 1
        elif Department == "Research & Development":
            Department = 2
        else:
            Department = 3
    with col4:
        DistanceFromHome = st.number_input('DistanceFromHome')

    with col1:
        Education = st.number_input('Education')
    
    with col2:
        EnvironmentSatisfaction = st.number_input('EnvironmentSatisfaction')

    with col3:
        Gender_Male = 0
        display = ("male", "female")
        options = list(range(len(display)))
        value = st.selectbox("Gender", options, format_func=lambda x: display[x])
        if value == "male":
            Gender_Male = 1
        elif value == "female":
            Gender_Male = 0

    with col4:
        HourlyRate = st.number_input('HourlyRate')

    with col1:
        JobInvolvement = st.number_input('JobInvolvement')

    with col2:
        JobLevel = st.number_input('JobLevel')

    with col3:
        JobRole = st.selectbox("JobRole", ["Sales Representative",
        "Sales Executive",
        "Healthcare Representative",
        "Laboratory Technician",
        "Research Scientist",
        "Research Director",
        "Manufacturing Director",
        "Manager",
        "Human Resources"])
        if JobRole == "Sales Representative":
            JobRole = 1
        if JobRole == "Sales Executive":
            JobRole = 2
        if JobRole == "Healthcare Representative":
            JobRole = 3
        if JobRole == "Laboratory Technician":
            JobRole = 4
        if JobRole == "Research Scientist":
            JobRole = 5
        if JobRole == "Research Director":
            JobRole = 6
        if JobRole == "Manufacturing Director":
            JobRole = 7
        if JobRole == "Manager":
            JobRole = 8
        else:
            JobRole = 9
         

    with col4:
       JobSatisfaction = st.number_input('JobSatisfaction')

    with col1:
        MaritalStatus  = st.selectbox("MaritalStatus", ["Divorced", "Single", "Married"])
        if Department == "Single":
            MaritalStatus = 1
        elif Department == "Married":
            MaritalStatus = 2
        else:
            MaritalStatus = 0

    with col2:
        MonthlyIncome = st.number_input('MonthlyIncome')

    with col3:
        MonthlyRate = st.number_input('MonthlyRate')

    with col4:
        NumCompaniesWorked = st.number_input('NumCompaniesWorked')

    with col1:
        OverTime = st.selectbox("OverTime", ["yes", "no"])
        OverTime = 1 if OverTime == "yes" else 0

    with col2:
        PercentSalaryHike = st.number_input('PercentSalaryHike')

    with col3:
        PerformanceRating  = st.number_input('PerformanceRating ')

    with col4:
        RelationshipSatisfaction = st.number_input('RelationshipSatisfaction')

    with col1:
        StockOptionLevel = st.number_input('StockOptionLevel')

    with col2:
        TotalWorkingYears = st.number_input('TotalWorkingYears')

    with col3:
        TrainingTimesLastYear = st.number_input('TrainingTimesLastYear')

    with col4:
        WorkLifeBalance = st.number_input('WorkLifeBalance')

    with col1:
        YearsAtCompany = st.number_input('YearsAtCompany')

    with col2:
        YearsInCurrentRole = st.number_input('YearsInCurrentRole')

    with col3:
        YearsSinceLastPromotion = st.number_input('YearsSinceLastPromotion')

    with col4:
        YearsWithCurrManager = st.number_input('YearsWithCurrManager')


    # code for Prediction
    Attrition_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Employee Attrition Prediction Result"):

        user_input = [Age, DailyRate, Department, DistanceFromHome, Education, EnvironmentSatisfaction, Gender_Male,
                       HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate, 
                       NumCompaniesWorked, OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StockOptionLevel, 
                       TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, 
                       YearsSinceLastPromotion, YearsWithCurrManager]

        prediction = Attrition_model.predict([user_input])

        if prediction[0] == 1:
            Attrition_diagnosis = "The employee leaving the organizations for an reason"
            # image = Image.open('positive.jpg')
            # st.image(image, caption='')
        else:
            Attrition_diagnosis = "The employee stayed in the organizations"
            # image = Image.open('negative.jpg')
            # st.image(image, caption='')
    st.success('Hi'+' '+name+' , ' + Attrition_diagnosis)


#Employee Promotion Prediction
if selected == 'Employee Promotion Prediction':
    
    st.title(":red[Employee Promotion Prediction using ML]")
    # image = Image.open('Kidney.jpg')
    # st.image(image, caption='Kidney disease')
    name = st.text_input("Name:")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        Age = st.number_input('Age', min_value=18, max_value=65, step=1)

    with col2:
        DailyRate = st.number_input('DailyRate',min_value=102, max_value=1496, step=1)

    with col3:
        HourlyRate = st.number_input('HourlyRate',min_value=30, max_value=100, step=1)

    with col4:
        DistanceFromHome = st.number_input('DistanceFromHome',min_value=1, max_value=29, step=1)

    with col1:
        MonthlyIncome = st.number_input('MonthlyIncome',min_value=1009, max_value=19999, step=1)

    with col2:
        MonthlyRate = st.number_input('MonthlyRate',min_value=2094, max_value=26999, step=1)

    with col3:
        PercentSalaryHike = st.number_input('PercentSalaryHike',min_value=11, max_value=25, step=1)
         
    with col4:
        EmployeeNumber = st.number_input('EmployeeNumber',min_value=1, max_value=2068, step=1)

    with col1:
        TotalWorkingYears = st.number_input('TotalWorkingYears',min_value=0, max_value=40, step=1)

    with col2:
        YearsAtCompany = st.number_input('YearsAtCompany',min_value=0, max_value=40, step=1)

    user_input = np.array([Age, YearsAtCompany, TotalWorkingYears, DailyRate, PercentSalaryHike, DistanceFromHome, 
                           MonthlyRate, MonthlyIncome, EmployeeNumber, HourlyRate])
    
    user_input = user_input.reshape(1, -1) 

    # creating a button for Prediction    
    if st.button(" Employee Promotion Prediction Result"):
        prediction = Promotion_model.predict(user_input)
        # rounded_prediction = round(prediction[0])
        st.success(f"the Employee {name} got Promotion by {prediction[0]:,.1f} years") 
        # prediction[0]:,.1f