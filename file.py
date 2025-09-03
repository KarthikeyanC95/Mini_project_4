import os
import pickle
import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import joblib

st.set_page_config(layout="wide")

watermark_css = """
<style>
    .watermark {
        position: fixed;
        bottom: 10px;
        right: 10px;
        font-size: 14px;
        color: gray;
        opacity: 0.7;
    }
</style>
<div class="watermark"> Data Scientist Karthikeyan</div>
"""


st.markdown(watermark_css, unsafe_allow_html=True)

# Custom CSS for the watermark


st.markdown(
        """
        <style>
            div.stButton > button {
                width: 100%; /* Full Width Inside Column */
                height: 50px;
                font-size: 18px;
                font-weight: bold;
                color: #0047AB;  /* Ocean Blue Text */
                text-align: center;
                background: white; /* White Background */
                border: 2px solid #0047AB; /* Blue Border */
                border-radius: 8px;
                cursor: pointer;
                transition: 0.3s;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            }

            div.stButton > button:hover {
                background: #0047AB; /* Blue Background on Hover */
                color: white; /* White Text on Hover */
            }
        </style>
        """,
        unsafe_allow_html=True
    )

#for button backgroun
st.markdown(
        """
        <style>
            div.stButton > button {
                width: 100%; /* Full Width Inside Column */
                height: 50px;
                font-size: 18px;
                font-weight: bold;
                color: #0047AB;  /* Ocean Blue Text */
                text-align: center;
                background: white; /* White Background */
                border: 2px solid #0047AB; /* Blue Border */
                border-radius: 8px;
                cursor: pointer;
                transition: 0.3s;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            }

            div.stButton > button:hover {
                background: #0047AB; /* Blue Background on Hover */
                color: white; /* White Text on Hover */
            }
        </style>
        """,
        unsafe_allow_html=True
    )




# Initialize session state
if "page" not in st.session_state:
    st.session_state["page"] = "Home"
if "subpage" not in st.session_state:
    st.session_state["subpage"] = None
	
# Navigation functions
def navigate_to(page_name, subpage_name=None):
    st.session_state["page"] = page_name
    st.session_state["subpage"] = subpage_name

if st.session_state["page"] == "Home":
    st.markdown(
    """
    <style>
        .unique-title {
            text-align: center;
            font-size: 50px;
            font-weight: bold;
            text-transform: uppercase;
            color: #0047AB; /* Ocean Blue */
            white-space: nowrap; /* Ensures it's always on one line */
            overflow: hidden;
            text-overflow: ellipsis;
            background: linear-gradient(to right, #0047AB, #007BFF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 2px;
            position: relative;
            top: -20px; /* Moves text higher */
        }
    </style>
    <div class='unique-title'>EMPLOYEE ATTRITION ANALYSIS AND PREDICTION</div>
    """,
    unsafe_allow_html=True
)




    # st.image(r'https://th.bing.com/th/id/OIP.XYqwtumU1KttgweY4tInCQHaEf?w=1000&h=607&rs=1&pid=ImgDetMain')
    col1, col2 = st.columns(2)
    image_size = 150

    with col1:
        st.image(
            r"https://www.r-exercises.com/wp-content/uploads/2016/11/Selecting-a-Real-Estate-Agent-Red.png",
            use_container_width=True,
        )
        if st.button("Predicting Employee Attrition", use_container_width=True):
            navigate_to("Predicting Employee Attrition")
        

    with col2:
        st.image(
            r"https://th.bing.com/th/id/OIP.F_v2ZXGhy9OlR_QbYA-NXgHaHa?pid=ImgDet&w=170.41420118343197&h=180&c=7&dpr=1.3",
            use_container_width=True,
        )
        if st.button("Predicting Performance Rating", use_container_width=True):
            navigate_to("Predicting Performance Rating")

#Employee Attrition Prediction
if st.session_state["page"] == "Predicting Employee Attrition":
    st.markdown(
    """
    <style>
        .eda-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            text-transform: uppercase;
            white-space: nowrap; /* Ensures it's always on one line */
            overflow: hidden;
            text-overflow: ellipsis;
            position: relative;
            top: -10px; /* Moves text slightly up */
            letter-spacing: 1px;
            background: linear-gradient(to right, #0047AB, #007BFF); /* Ocean Blue Gradient */
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
    <div class='eda-title'>Employee Attrition Prediction using ML</div>
    """,
    unsafe_allow_html=True
    )

    st.markdown(
    "<h4 style='text-align: center;'>Predict whether an employee will leave the company (attrition).</h4>",
    unsafe_allow_html=True
    )
    
	  
	
    st.write("Press the button to go back to Home/Previous Page:")  
    col3,col4=st.columns(2)
    with col3:
      if st.button("🔙 Back", use_container_width=True):
            navigate_to("Home")
    with col4:
      if st.button("🏠 Home", use_container_width=True):
            navigate_to("Home")

    # Load the trained models
    decisionforemployeeattri = joblib.load(r'/content/drive/MyDrive/Employee Attrition Analysis and Prediction/_Models_decisionforemployeeattri.pkl')
    ohe_for_emp_att = joblib.load(r'/content/drive/MyDrive/Employee Attrition Analysis and Prediction/_Models_onehotencoderforemp_attri.pkl')
    le_for_emp_att = joblib.load(r'/content/drive/MyDrive/Employee Attrition Analysis and Prediction/_Models_labelforemployeeattri.pkl')

        

    st.header("Enter Employee Details")

    # User input fields
    # First row
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=65, step=1)
    with col2:
        handled_MonthlyIncome = st.number_input("Monthly Income", min_value=1000, step=500)
    with col3:
                JobSatisfaction = st.slider("Job Satisfaction", min_value=1, max_value=5, step=1)

    # Second row
    col1, col2, col3 = st.columns(3)
    with col1:
        YearsAtCompany = st.number_input("Years at Company", min_value=0, step=1)
    with col2:
        overtime = st.selectbox("Overtime", ['Yes', 'No'])
        
    with col3:
        NumCompaniesWorked = st.number_input("Number of Companies Worked", min_value=0, step=1)

    # Third row (Categorical)
    col1, col2 = st.columns(2)
    with col1:
        Department = st.selectbox("Department", ['Sales', 'HR', 'Research & Development'])
    with col2:
        MaritalStatus = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])


    # Convert to DataFrame
    le_ot = le_for_emp_att.transform(np.array([overtime]).reshape(1, -1)).flatten()[0]  
    user_data = pd.DataFrame({'Department': [Department], 'MaritalStatus': [MaritalStatus]})
    encoded_user_input = ohe_for_emp_att.transform(user_data)

    # Convert numerical inputs to NumPy array
    num_inputs = np.array([age, handled_MonthlyIncome, JobSatisfaction, YearsAtCompany, le_ot, NumCompaniesWorked]).reshape(1, -1)

    # Concatenate numerical and categorical features
    combined_input = np.concatenate([num_inputs, encoded_user_input], axis=1)

    # Prediction
    if st.button("Predict Attrition"):
        employee_attrition = decisionforemployeeattri.predict(combined_input)[0]
        
        prediction_text = "Likely to Leave" if employee_attrition == 'Yes' else "Likely to Stay"
        color = "green" if employee_attrition == 'No' else "red"
        st.markdown(f"### <span style='color:{color};'>Employee Attrition Prediction: {prediction_text}</span>", unsafe_allow_html=True)
    
    # Visualization
        st.subheader("Employee Attributes")
        st.bar_chart(pd.DataFrame({
            "Feature": ["Age", "Monthly Income", "Job Satisfaction", "Years at Company", "Overtime", "Companies Worked"],
            "Value": [age, handled_MonthlyIncome, JobSatisfaction, YearsAtCompany, le_ot, NumCompaniesWorked]
        }).set_index("Feature")) 


#Employee Promotion Prediction
elif st.session_state.get("page") == "Predicting Performance Rating":
    
    st.markdown(
        """
        <style>
            .eda-title {
                text-align: center;
                font-size: 36px;
                font-weight: bold;
                text-transform: uppercase;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                position: relative;
                top: -10px;
                letter-spacing: 1px;
                background: linear-gradient(to right, #0047AB, #007BFF);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
        </style>
        <div class='eda-title'>Employee Performance Prediction using ML</div>
        """,
        unsafe_allow_html=True
    )
	
	  
			
    st.write("Press the button to go back to Home/Previous Page:")
    col3,col4=st.columns(2)
    with col3:
      if st.button("🔙 Back", use_container_width=True):
        navigate_to("Home")
    with col4:
      if st.button("🏠 Home", use_container_width=True):
        navigate_to("Home")
    


    ohe_for_job_sat = joblib.load(r'/content/drive/MyDrive/Employee Attrition Analysis and Prediction/_Models_onehotforjobsat.pkl')
    label_for_job_sat = joblib.load(r'/content/drive/MyDrive/Employee Attrition Analysis and Prediction/_Models_labelforjobsat.pkl')
    logistic_model = joblib.load(r'/content/drive/MyDrive/Employee Attrition Analysis and Prediction/_Models_logistic_for_performance_rating.pkl')

# Page title

        
    # ---------- Input Form ----------
    st.write("Fill in the details to predict the employee's performance rating:")
    with st.form("perf_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            Age = st.number_input("Age", min_value=18, max_value=60, step=1)
            DistanceFromHome = st.number_input("Distance from Home", step=1)
            Education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
            EnvironmentSatisfaction = st.slider("Environment Satisfaction", 1, 4)
            JobInvolvement = st.slider("Job Involvement", 1, 4)
            MonthlyIncome = st.number_input("Monthly Income", step=100)
            
        with col2:
            MonthlyRate = st.number_input("Monthly Rate", step=100)
            PercentSalaryHike = st.slider("Percent Salary Hike", 0, 100)
            RelationshipSatisfaction = st.slider("Relationship Satisfaction", 1, 4)
            WorkLifeBalance = st.slider("Work-Life Balance", 1, 4)
            YearsInCurrentRole = st.number_input("Years in Current Role", step=1)
            OverTime = st.selectbox("OverTime", ["Yes", "No"])
            
        with col3:
        # Categorical inputs
            BusinessTravel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Frequently", "Travel_Rarely"])
            Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            EducationField = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
            Gender = st.selectbox("Gender", ["Male", "Female"])
            JobRole = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", 
                                            "Healthcare Representative", "Manager", "Sales Representative", 
                                            "Research Director", "Human Resources"])
            MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        submitted = st.form_submit_button("Predict")

        if submitted:
        # Numerical inputs
            numerical_inputs = [
                Age,
                DistanceFromHome,
                Education,
                EnvironmentSatisfaction,
                JobInvolvement,
                MonthlyIncome,
                MonthlyRate,
                PercentSalaryHike,
                RelationshipSatisfaction,
                WorkLifeBalance,
                YearsInCurrentRole,
                label_for_job_sat.transform([[OverTime]])[0],
                
            ]

        # Categorical inputs
            cat_input = pd.DataFrame([{
            "BusinessTravel": BusinessTravel,
            "Department": Department,
            "EducationField": EducationField,
            "Gender": Gender,
            "JobRole": JobRole,
            "MaritalStatus": MaritalStatus
            }])

            cat_encoded = ohe_for_job_sat.transform(cat_input)

        # Final input
            final_input = np.concatenate([np.array(numerical_inputs).reshape(1, -1), cat_encoded], axis=1)

            prediction = logistic_model.predict(final_input)[0]

# Map prediction to label
            rating_label = {
    3: "Excellent (Exceeds Expectations)",
    4: "Outstanding (Top Performer)"
    }.get(prediction, "Unknown")

# Set background gradient by rating
            bg_gradient = "linear-gradient(to right, #3A7BD5, #00d2ff);" if prediction == 3 else "linear-gradient(to right, #8E2DE2, #4A00E0);"

# Styled block with CSS
            st.markdown(f"""
    <div style="
        padding: 1.5rem;
        border-radius: 15px;
        background: {bg_gradient};
        color: white;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.15);
    ">
         Predicted Performance Rating: {prediction} - {rating_label}
    </div>
""", unsafe_allow_html=True)
