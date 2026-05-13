import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import pickle
import os    
import shap
import matplotlib.pyplot as plt

def load_model():
    """
    Load the pre-trained model from a pickle file.
    """
    model_path = os.path.join("model", "model_lgbm_rekurze.model")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()


# explainer jen jednou po loadovani modelu
explainer = shap.TreeExplainer(model)

def risk_assessment_page():
    st.markdown(
        """
        <style>
        .block-container {
            /* Keep content below Streamlit top bar */
            padding-top: calc(3.25rem + env(safe-area-inset-top, 0px));
        }
        .app-header {
            margin-bottom: 0.5rem;
        }
        .app-header h1 {
            margin: 0 0 0.2rem 0;
            line-height: 1.15;
            font-size: clamp(2rem, 7.2vw, 3rem);
            font-weight: 800;
        }
        .app-header p {
            margin: 0;
            font-size: 0.9rem;
            color: #6b7280;
        }
        div[data-baseweb="tag"] {
            min-height: 2rem;
        }
        div[data-testid="stCheckbox"] label {
            padding-top: 0.1rem;
            padding-bottom: 0.1rem;
        }
        /* Keep TNM columns side-by-side even on phones */
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: nowrap !important;
            gap: 0.4rem !important;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            min-width: 0 !important;
            flex: 1 1 0 !important;
        }
        </style>
        <div class="app-header">
            <h1>Breast Cancer Recurrence Prediction</h1>
            <p>Risk Assessment Settings -- this tool is intended for use by doctors (of data science) only.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Controls
    st.markdown(
        "#### Age", 
    )    
    age = st.slider("Select your current age", 20, 90, 45)
    
    def age_normalization(age):
        """
        code for age normalization:
        102 - 20-29 let
        103 - 30-39 let
        104 - 40-49 let
        105 - 50-59 let
        106 - 60-69 let
        107 - 70-79 let
        108 - 80-89 let
        109 - 90+ let
        
        """
        if age < 30:
            return 102
        elif age < 40:
            return 103
        elif age < 50:
            return 104
        elif age < 60:
            return 105
        elif age < 70:
            return 106
        elif age < 80:
            return 107
        elif age < 90:
            return 108
        else:
            return 109
        
    vekova_kategorie_10let_dg = age_normalization(age)

    print(f"Selected age: {age}")

    # Number of hospitalizations during primary therapy 
#    st.write("#### Number of Hospitalizations During Primary Therapy")
    st.markdown(
        "#### Number of Hospitalizations During Primary Therapy", 
        help="Select the number of hospitalizations during primary therapy."
    )

#   st.write("Select the number of hospitalizations during primary therapy")
    
    # Number input
    pl_pocet_hp = st.number_input("Number of hospitalizations", min_value=0, max_value=50, value=3, step=1)
    print(f"Selected number of hospitalizations: {pl_pocet_hp}")
    
    st.write("#### TNM Classification")
    t_col, n_col, m_col = st.columns(3, gap="small")
    with t_col:
        st.markdown("**T**")
        t_useless = st.selectbox(
            "T selector",
            ['0','1','1a','1a2','1b','1c','1m','2','2a', '2b', '2c', '3', '3b','4','4a','4b','4c','4d','X','a','is','isD','isL','isP'],
            index=1,
            label_visibility="collapsed",
        )
    with n_col:
        st.markdown("**N**")
        tnm_klasifikace_n_kod = st.selectbox(
            "N selector",
            ['0', '1', '1a', '1b', '1c', '1m', '2', '2a', '2b', '2c', '3', '3a', '3b', '3c', 'X'],
            index=1,
            label_visibility="collapsed",
        )
    with m_col:
        st.markdown("**M**")
        m_useless = st.selectbox(
            "M selector",
            ['0', '1', '1d', 'X'],
            index=0,
            label_visibility="collapsed",
        )

    # Create one-hot encoding for n_classification ['0', '1', '1a', '1b', '1c', '1m', '2', '2a', '3a', 'X']
    n_categories = ['0', '1', '1a', '1b', '1c', '1m', '2', '2a', '3a', 'X']
    #tnm_klasifikace_n_kod = [1 if n_classification == cat else 0 for cat in n_categories]

    # For debugging/development purposes
    print(f"Selected N classification: {tnm_klasifikace_n_kod}")
    # print(f"One-hot encoding: {tnm_klasifikace_n_kod}")

    exam_options = ["CT", "MAMO", "MRI", "PET-CT", "RTG", "SCINT", "SONO", "SPECT", "OTHER"]

    # Primary diagnosis section
    st.subheader("Primary Diagnosis Examinations")
    st.write("Select all examinations performed at diagnosis.")
    primary_exams = st.multiselect(
        "Primary diagnosis examinations",
        options=exam_options,
        default=["MAMO", "SCINT", "OTHER"],
        label_visibility="collapsed",
        key="primary_exams_multiselect",
    )
    pl_vysetreni_MAMO = "MAMO" in primary_exams
    pl_vysetreni_SCINT = "SCINT" in primary_exams
    pl_vysetreni_OTHER = "OTHER" in primary_exams

    # Post-therapy section
    st.subheader("Post-Therapy Examinations")
    st.write("Select all examinations performed after diagnosis.")
    post_exams = st.multiselect(
        "Post-therapy examinations",
        options=exam_options,
        default=["MRI", "SPECT"],
        label_visibility="collapsed",
        key="post_exams_multiselect",
    )
    pd_vysetreni_SPECT = "SPECT" in post_exams


    st.write("#### Last Hospitalization Reason During Primary Therapy")
    st.write("Select the last reason for hospitalization during primary therapy:")

    # {"09-I06-04":"Single breast resection including lymph node removal in CVSP in patients with CC=0","09-I06-05":"Breast resection including lymph node removal outside CVSP","09-I09-02":"Single breast resection for malignant neoplasm in CVSP in patients with CC=0","09-I09-03":"Breast resection for malignant neoplasm outside CVSP", 'OTHER': 'OTHER'}
    # Centrum vysoce specializované péče in english: "Highly specialized care center"
    pl_hp_drg_posledni_dict = {
        "09-I06-04": "Single breast resection including lymph node removal in highly specialized care center in patients with CC0",
        "09-I06-05": "Breast resection including lymph node removal outside highly specialized care center",
        "09-I09-02": "Single breast resection for malignant neoplasm in highly specialized care center in patients with CC0",
        "09-I09-03": "Breast resection for malignant neoplasm outside highly specialized care center",
        'OTHER': 'OTHER'
    }


    pl_hp_drg_posledni = st.selectbox("Last hospitalization reason", options=[
        "09-I06-04",
        "09-I06-05",
        "09-I09-02",
        "09-I09-03",
        'OTHER'
    ], format_func=lambda x: pl_hp_drg_posledni_dict.get(x, x), index=4, label_visibility="collapsed")

    print(f"Selected last hospitalization reason: {pl_hp_drg_posledni}")

    st.write("#### All kinds of therapy")
    st.write("Please select the types of primary therapy in the order they were performed")

    # "pl_typ_lecby_1" ['C', 'H', 'I', 'O', 'R', 'T']
    # Terapie všeho druhu in english: "All kinds of therapy"
    pl_typy_lecby_dict = {
        'C': 'Chemotherapy',
        'H': 'Hormone therapy',
        'I': 'Immunotherapy',
        'O': 'Surgery',
        'R': 'Radiotherapy',
        'T': 'Central anti-tumor treatment'
    }


    pl_typy_lecby = st.multiselect("All kinds of therapy", options=[
        'C', 'H', 'I', 'O', 'R', 'T'
    ], format_func=lambda x: pl_typy_lecby_dict.get(x, x), default=['C', 'H', 'R'], label_visibility="collapsed")

    if pl_typy_lecby != []:
        pl_typ_lecby_1 = pl_typy_lecby[0]
        print(f"Selected therapy type: {pl_typ_lecby_1}")
    else:
        pl_typ_lecby_1 = None # TODO: Handle this case

    if 'C' in pl_typy_lecby:        
        je_pl_chemo = 1
    else:
        je_pl_chemo = 0

    if 'H' in pl_typy_lecby:
        je_pl_hormo = 1
    else:
        je_pl_hormo = 0

    if 'R' in pl_typy_lecby:
        je_pl_radio = 1
    else:
        je_pl_radio = 0
    
    print(pl_typy_lecby)
    print(f"Selected chemotherapy: {je_pl_chemo}")
    print(f"Selected hormone therapy: {je_pl_hormo}")
    print(f"Selected radiotherapy: {je_pl_radio}")

    print({'vekova_kategorie_10let_dg': vekova_kategorie_10let_dg,
        'je_pl_hormo': je_pl_hormo,
        'je_pl_chemo': je_pl_chemo,
        'pl_pocet_hp': pl_pocet_hp,
        'tnm_klasifikace_n_kod': tnm_klasifikace_n_kod,
        'je_pl_radio': je_pl_radio,
        'pd_spect': pd_vysetreni_SPECT,
        'pl_mamo': pl_vysetreni_MAMO,
        'pl_hp_drg_posledni': pl_hp_drg_posledni,
        'pl_jina': pl_vysetreni_OTHER,
        'pl_scint': pl_vysetreni_SCINT,
        'pl_typ_lecby_1': pl_typ_lecby_1})

    row_to_predict = pd.DataFrame({
        'vekova_kategorie_10let_dg': vekova_kategorie_10let_dg,
        'je_pl_hormo': je_pl_hormo,
        'je_pl_chemo': je_pl_chemo,
        'pl_pocet_hp': pl_pocet_hp,
        'tnm_klasifikace_n_kod': tnm_klasifikace_n_kod,
        'je_pl_radio': je_pl_radio,
        'pd_spect': pd_vysetreni_SPECT,
        'pl_mamo': pl_vysetreni_MAMO,
        'pl_hp_drg_posledni': pl_hp_drg_posledni,
        'pl_jina': pl_vysetreni_OTHER,
        'pl_scint': pl_vysetreni_SCINT,
        'pl_typ_lecby_1': pl_typ_lecby_1
    }, index=[0])

    print(f"Row to predict: {row_to_predict}")

    # Boolean values to 0 or 1
    row_to_predict['pd_spect'] = int(row_to_predict['pd_spect'])
    row_to_predict['pl_mamo'] = int(row_to_predict['pl_mamo'])
    row_to_predict['pl_jina'] = int(row_to_predict['pl_jina'])
    row_to_predict['pl_scint'] = int(row_to_predict['pl_scint'])
    
    row_to_predict['tnm_klasifikace_n_kod'] = row_to_predict['tnm_klasifikace_n_kod'].astype('category').cat.set_categories(n_categories)
    row_to_predict['pl_hp_drg_posledni'] = row_to_predict['pl_hp_drg_posledni'].astype('category').cat.set_categories(pl_hp_drg_posledni_dict.keys())
    row_to_predict['pl_typ_lecby_1'] = row_to_predict['pl_typ_lecby_1'].astype('category').cat.set_categories(pl_typy_lecby_dict.keys()) 

    print(f"Row to predict after conversion: {row_to_predict}")

 
    # Main content
    st.write("## Risk Assessment Tool")
    st.write("Review your selections and tap predict to calculate recurrence risk.")

    # Predict button
    if st.button("Predict Recurrence"):
        result = model.predict(row_to_predict)
        st.write(f"##### Your recurrence risk is: {100* result[0]:.2f}%")
        explanation = explainer(row_to_predict)  # data je jeden radek
        shap.force_plot(
            -0.25815062496723745,
            explanation[0].values,
            features=row_to_predict.values[0],
            feature_names=list(row_to_predict.columns),
            show=False,
            link="logit",
            matplotlib=True,
            text_rotation = 45,
        )

        fig = plt.gcf()
        st.pyplot(fig, use_container_width=True)

        # st.write("### Prediction Results")

        # # Example SHAP plot
        # shap_data = pd.DataFrame({"Feature": ["Age", "Tumor Size", "Grade"], "Importance": [0.4, 0.35, 0.25]})
        # fig = px.bar(shap_data, x="Importance", y="Feature", orientation="h", title="SHAP Feature Importance")
        # st.plotly_chart(fig)

        # # Example wider plot
        # x = np.linspace(0, 10, 100)
        # y = np.sin(x)
        # fig = px.line(x=x, y=y, title="Example Wider Plot")
        # st.plotly_chart(fig, use_container_width=True)