import Preprocessing as pp
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import seaborn as sns
import time
import plotly.figure_factory as ff
import plotly.graph_objects as go


def distplot(x, predicted, requested):
    hist_data = [x]
    group_labels = ['distplot'] # name of the dataset

    fig = ff.create_distplot(hist_data, group_labels, bin_size=0.1)

    fig.update_layout(
        xaxis_title="Funding Amount (x € 1 million)",
        yaxis_title="Density",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        ),
        showlegend=False
    )
    fig.add_shape(
        # Line Vertical
        go.layout.Shape(
            type="line",
            x0=predicted,
            y0=0,
            x1=predicted,
            y1=1.2,
            line=dict(
                color="Orange",
                width=2
            )
        )
    )

    st.plotly_chart(fig, height=600, width=800)

def open_data(testtrain='train'):
    with open("dataset_" + testtrain, "rb") as file:
        return pickle.load(file)

@st.cache(show_spinner = False)
def load_models():
    with open("models", "rb") as file:
        print('test')
        return pickle.load(file)

def load_features():
    with open("features", "rb") as file:
        return pickle.load(file)

def split_X_and_Y(dataset, prediction_task='last_funding_regression'):
    """
    Since we're trying to do a few different prediction tasks, I thought to create this function that neatly makes
    the right X and Y datasets. Input the prediction task:
    'last_funding_regression' -- For the regression task that predicts how much funding a startup will get.
    'funding_stage_classification' -- For predicting the type of funding round the startup will get (seed, series A, etc.)
    'growth_stage_classification' -- For predicting what growth stage the startup is in, as suggested by Sarah's father.
    """
    
    last_funding_features = [col for col in dataset.columns if col.startswith('last_funding')]
    last_funding_round_features = [col for col in dataset.columns if col.startswith('last_funding_round_round')]
    growth_stage_features = [col for col in dataset.columns if col.startswith('growth_stage')]

    X = dataset.drop(last_funding_features+growth_stage_features, axis=1)
        
    if prediction_task == 'last_funding_regression':
        Y = dataset['last_funding']        
    elif prediction_task == 'funding_stage_classification':
        Y = dataset[last_funding_round_features]
    elif prediction_task == 'growth_stage_classification':
        Y = dataset[growth_stage_features]
        
    return X, Y

def get_empty_result(dataset, features):
    final_result = dataset[features].iloc[0:1]

    final_result.index = [""]

    for col in final_result.columns:
        final_result[col] = 0
    return final_result

def format_multiselect(uglyname):
    uglyname = uglyname.replace("categories_", "")
    return uglyname.title()

def format_clients(uglyname):
    uglyname = uglyname.replace("client_focus_", "")
    return uglyname.title()

def format_revenue(uglyname):
    uglyname = uglyname.replace("revenues_", "")
    if uglyname == 'manufacturing':
        return "Sale"
    elif uglyname == 'commission':
        return 'Service Fee'
    return uglyname.title()

def format_serial(uglyname):
    uglyname = uglyname.replace("company_founders_is_serial_", "")
    return uglyname.title()

def format_classes(uglynames):
    result = []
    for uglyname in uglynames:
        goodname = uglyname.replace("last_funding_round_round_", "")
        result.append(goodname.title())
    return result

def show_general_questions(result):
    result['employees_date_last'] = 2019

    st.write("## What category are you active in?")
    selected_cats = st.multiselect(label='categories', options=categories, format_func=format_multiselect)

    for cat in selected_cats:
        result[cat] = 1

    st.write("## What year did you make your first sale?")
    launch_date = st.slider("Year", 2010, 2020, key='whatereverer')

    result['launch_date'] = launch_date

    st.write("## Who are your customers?")
    selected_client_focus = st.selectbox(label="Business or Consumer", options=client_focus, format_func=format_clients)

    result[selected_client_focus] = 1

    st.write("## What type of revenue do you gather?")
    revenue_model = st.selectbox(label="Revenue model", options=revenue, format_func=format_revenue)

    result[revenue_model] = 1    

    st.write("## Do you or your co-founders have previous experience in founding startups?")
    selected_serial = st.selectbox(label="Yes or No", options=serial, format_func=format_serial)

    result[selected_serial] = 1

    st.write("## How many founders did your company start out with?")
    past_founders = st.slider("# of founders", 1, int(max(X_train["company_founders_statuses_past"])), 1)

    result["company_founders_statuses_past"] = past_founders

    st.write("## How many of those are still active in the company?")
    current_founder = st.slider("# of founders", 0, int(past_founders), 1)

    result["company_founders_statuses_current"] = current_founder

    st.write("## How many people are currently working at the company?")
    employees = st.slider("# number of employees", 0, 250)

    result['employees_values_last'] = employees

    return result

def show_previous_funding_questions(result):
    st.write("## What is the total amount of funding you have received so far?")
    total_funding = st.slider("Total Funding ( x € 1000)", 0.0, 5000.0)/1000

    result['total_funding_report'] = total_funding

    st.write("## What year did you first receive funding?")
    funding_year = st.slider("Year", 2010, 2019)

    result['first_funding_round'] = funding_year
    result['seed_year'] = funding_year

    st.write("## How many funding rounds have you previously had in total?")
    no_fundings = st.slider("# of rounds, excluding upcoming/current", 1, 5)

    result['fundings_total'] = no_fundings

    st.write("## What year did you most recently receive funding?")
    kpi_year = st.slider("Year", 2010, 2019, key='dummykeywhatever')

    result['kpi_valuation_date'] = kpi_year

    st.write("## What was your valuation at that time?")
    kpi_value = st.slider("Valuation (x million €'s)", 0.0, 50.0)

    result['kpi_valuation_eur'] = kpi_value*1e6

    return result

def show_vacancy_questions(result, vacancies):
    st.write("## How many of those are for these specific positions?")
    engineers = st.slider("# of engineer vacancies", 0, vacancies)
    management = st.slider("# of management vacancies", 0, vacancies)
    sales = st.slider("# of sales positions", 0, vacancies)

    result["jobs_engineering"] = engineers
    result["jobs_management"] = management
    result["jobs_sales"] = sales

    return result

def show_expectation_questions():
    st.write("# How much funding are you looking for?")
    request = st.slider("Funding (x € 1000)", 100, 2000)*1000

    types = ['Pre-Seed', 'Seed', 'Series A', 'Series B']

    st.write("# What round of funding do you consider yourself to be looking for?")
    fround = st.selectbox(label='Funding round', options=types)
    return request, fround

def show_prediction(result, criteria):    
    classification_prediction = classifier.predict(result)
    regression_prediction = float(regressor.predict(result))

    pred_class = classes_formatted[np.argmax(classification_prediction)]

    expected = criteria[0]
    fround = criteria[1]
    
    round_made = False
    if pred_class == 'Acquisition' or fround == "Pre-Seed":
        round_made = True

    elif pred_class == fround:
        round_made = True

    amount_made = False
    if expected < regression_prediction*1000000:
        amount_made = True

    if round_made and amount_made:
        st.balloons()
        st.write("# You seem ready for funding!")
        st.write("### Ventures at your level can expect to receive: € " + str(int(regression_prediction*1e6))
         + ",- of funding in their next round.")

        st.write("### Ventures with your characteristics commonly succesfully apply for " + fround + " funding.")
    
    elif amount_made:
        st.write("# You don't seem quite ready for this type of funding!")
        st.write("### Ventures with your characteristics commonly succesfully apply for " + pred_class + 
            " funding, but " + fround + " funding is unlikely.")

        st.write("### However, ventures at your level can sometimes expect to receive up to: € " + str(int(regression_prediction*1e6))
         + ",- of funding in their next round, so your expectations on that front seem realistic.")

    elif round_made:
        st.write("# You don't seem quite ready for this type of funding!")
        st.write("### Ventures with your characteristics commonly succesfully apply for " + fround + " funding.")

        st.write("### However, your requested amount of € " + str(expected) +",- seems too high to be realistic. " 
            + "We generally expect ventures at your level to apply for up to € " + str(int(regression_prediction*1e6)) + ",-.")

    else:
        st.write("# You don't seem quite ready for this type of funding!")
        st.write("### Ventures with your characteristics commonly succesfully apply for " + pred_class + 
            " funding, but " + fround + " funding is unlikely.")

        st.write("### Moreover, your requested amount of € " + str(expected) +",- seems too high to be realistic. " 
            + "We generally expect ventures at your level to apply for up to € " + str(int(regression_prediction*1e6)) + ",-.")

    st.write("### The orange line in the graph below shows how you stack up to the other startups in our dataset, in terms of expected funding amount.")

    distplot(regressor.predict(test_data[features]), regression_prediction, expected)


train_data = open_data()
test_data = open_data('test')

X_train, y_train = split_X_and_Y(train_data, 'funding_stage_classification')

classes = y_train.columns[y_train.sum()>100]

classes_formatted = format_classes(classes)

classifier, regressor = load_models()
features = load_features()

result = get_empty_result(X_train, features)

st.title("Demo Investment Readiness Self-assessment")

categories = [feature for feature in features if feature.startswith('categories')]
client_focus = [feature for feature in features if feature.startswith('client_focus')]
revenue = [feature for feature in features if feature.startswith('revenues')]
serial = [feature for feature in features if feature.startswith('company_founders_is_serial')]
founders_past = [feature for feature in features if feature.startswith('company_founders_statuses_past')]
founders_current = [feature for feature in features if feature.startswith('company_founders_statuses_current')]

result = show_general_questions(result)

st.write("## Did you previously receive funding?")
previous = st.checkbox("Yes, I have previously received funding", False)

if previous:
    result = show_previous_funding_questions(result)

st.write("## How many outstanding vacancies do you have?")
vacancies = st.slider("# of vacancies", 0, 25)

result['jobs_Total'] = vacancies

if vacancies > 0:
    results = show_vacancy_questions(result, vacancies)

criteria = show_expectation_questions()

if st.button('Calculate'):
    with st.spinner("Calculating..."):
        show_prediction(result, criteria)




