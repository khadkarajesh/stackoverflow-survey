import numpy as np
from dash import Dash, html, dcc, Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go

survey_df = pd.read_csv("data.csv")
# survey_df.dropna(inplace=True)

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
sidebar = html.Div(
    [
        html.H2("Stackoverflow Survey", className="display-12"),
        html.Hr(),
        html.P(
            "Stackoverflow 2021 survey using dash", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("First line code written", href="/page-1", active="exact"),
                dbc.NavLink("Salary of Developers", href="/page-2", active="exact"),
                dbc.NavLink("Participants by Country", href="/page-3", active="exact"),
                dbc.NavLink("Participants by Developer Type", href="/page-4", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


def assign_value_to_age_range(x):
    if x == 'Younger than 5 years':
        return 0
    elif x == '5 - 10 years':
        return 1
    elif x == '11 - 17 years':
        return 2
    elif x == '18 - 24 years':
        return 3
    elif x == '25 - 34 years':
        return 4
    elif x == '35 - 44 years':
        return 5
    elif x == '45 - 54 years':
        return 6
    return 7


def get_first_code_age():
    age_first_code_df = survey_df.copy()

    age_first_code_grouped = age_first_code_df.groupby('Age1stCode').size().reset_index(name='size')
    total = age_first_code_grouped['size'].sum()
    age_first_code_grouped['percentage'] = age_first_code_grouped['size'].apply(lambda x: round(x / total * 100, 2))
    age_first_code_grouped['stared_code_index'] = age_first_code_grouped['Age1stCode'].apply(assign_value_to_age_range)
    age_first_code_grouped = age_first_code_grouped.sort_values(by='stared_code_index', ascending=False)

    fig = px.bar(age_first_code_grouped,
                 y="Age1stCode",
                 x="percentage",
                 text=age_first_code_grouped['percentage'],
                 orientation='h')

    fig.update_layout(xaxis_title="Percentage",
                      yaxis_title="Age",
                      title="First line of code written age percentage",
                      legend_title="Age vs Percentage")

    return html.Div([
        dcc.Graph(
            id='first_line_of_code_written',
            figure=fig
        )
    ])


def map_dev_type(x):
    if x == 'Academic researcher':
        return "Academic"
    elif x == "Data scientist or machine learning specialist":
        return "Data"
    elif x == "DevOps specialist":
        return "DevOps"
    elif x == "Database administrator":
        return "Db"
    elif x == "Developer, mobile":
        return "Mobile"
    elif x == "Developer, front-end":
        return "Frontend"
    elif x == "Developer, full-stack":
        return "Full-stack"
    elif x == "Developer, back-end":
        return "Backend"
    elif x == "Developer, QA or test":
        return "QA"
    elif x == "Developer, game or graphics":
        return "Game Dev"
    elif x == "Data or business analyst":
        return "Data/Biz"
    elif x == "Developer, embedded applications or devices":
        return "Developer"
    elif x == "Developer, desktop or enterprise applications":
        return "Developer"
    elif x == "Product manager":
        return "Manager"
    elif x == "System administrator":
        return "Sys"
    elif x == "Marketing or sales professional":
        return "sales"
    elif x == "Engineer, data":
        return "Data Eng"
    elif x == "Engineer, site reliability":
        return "SRE"
    elif x == "Other (please specify):":
        return "Others"
    elif x == "Senior Executive (C-Suite, VP, etc.)":
        return "Senior"
    elif x == 'Engineering manager':
        return "Engineering Manager"
    return x


def display_salary():
    salary_exp_df = survey_df.copy()

    salary_exp_df = salary_exp_df[salary_exp_df['YearsCodePro'].notna()]
    salary_exp_df = salary_exp_df[~salary_exp_df['YearsCodePro'].isin(['Less than 1 year', 'More than 50 years'])]
    salary_exp_df['YearsCodePro'] = salary_exp_df['YearsCodePro'].apply(lambda x: '0' if x == 'Less than 1 year' else x)
    salary_exp_df['YearsCodePro'] = salary_exp_df['YearsCodePro'].apply(
        lambda x: '50' if x == 'More than 50 years' else x)
    salary_exp_df['YearsCodePro'] = salary_exp_df['YearsCodePro'].astype('int')

    salary_exp_df["DevType"] = salary_exp_df["DevType"].str.split(";")
    salary_exp_df = salary_exp_df.explode("DevType").reset_index(drop=True)
    salary_exp_df = salary_exp_df.groupby(['DevType']).agg(
        {'ConvertedCompYearly': np.median, 'YearsCodePro': np.mean, 'ResponseId': 'size'}).reset_index()
    salary_exp_df.rename(columns={"ResponseId": "Participants"}, inplace=True)
    salary_exp_df['YearsCodePro'] = salary_exp_df['YearsCodePro'].apply(lambda x: round(x, 2))
    salary_exp_df['dev_type_mapper'] = salary_exp_df['DevType'].apply(map_dev_type)

    fig = px.scatter(salary_exp_df,
                     x="YearsCodePro",
                     y="ConvertedCompYearly",
                     color="Participants",
                     text='dev_type_mapper',
                     labels={
                         "YearsCodePro": "Average Years of Professional Experience",
                         "ConvertedCompYearly": "Median Yearly Salary(USD)",
                     })

    fig.update_traces(textposition='top center')
    fig.update_xaxes(nticks=10)
    fig.update_layout(
        height=1000,
        title_text='Salary and Experience by Developer type',
        yaxis=dict(
            range=[40000, 100000]
        ),
        xaxis=dict(
            dtick=0.5,
            # tick0=8.0
            range=[8.5, 16]
        )
    )
    return html.Div([
        dcc.Graph(
            id='salary_by_experience',
            figure=fig
        )
    ])


def display_participants():
    participants_df = survey_df.copy()
    participants_df = participants_df.groupby('Country').size().reset_index(name="size")
    data = dict(
        type='choropleth',
        locations=participants_df['Country'].values,
        locationmode='country names',
        z=participants_df['size'])
    fig = go.Figure(data=[data])
    fig.update_layout(height=600,
                      title={
                          'text': "Number of Participants by country",
                          'y': 0.9,
                          'x': 0.5,
                          'xanchor': 'center',
                          'yanchor': 'top'},
                      legend_title="Age vs Percentage")
    return html.Div([
        dcc.Graph(
            id='number_of_participants',
            figure=fig
        )
    ])


def get_dev_count():
    dev_type_count = dict()
    for i in range(len(survey_df)):
        types = survey_df['DevType'][i]
        if isinstance(types, str):
            dev_type = types.split(';')
            for j in dev_type:
                if j not in dev_type_count:
                    dev_type_count[j] = 1
                else:
                    dev_type_count[j] += 1
    return dev_type_count


def display_dev_count():
    dev_count_dict = get_dev_count()
    dev_count_df = pd.DataFrame()
    dev_count_df['dev'] = list(dev_count_dict.keys())
    dev_count_df['count'] = list(dev_count_dict.values())

    fig = px.treemap(dev_count_df, path=['dev'],
                     values='count')
    fig.update_traces(root_color="lightgrey")
    fig.update_layout(margin=dict(t=55, l=25, r=25, b=10))
    fig.update_layout(height=600,
                      title={
                          'text': "Participants by Developer Type",
                          'y': 0.9,
                          'x': 0.5,
                          'xanchor': 'center',
                          'yanchor': 'top'}, )

    return html.Div([
        dcc.Graph(
            id='dev_type_counts',
            figure=fig
        )
    ])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return html.P("This is the content of the home page!")
    elif pathname == "/page-1":
        return get_first_code_age()
    elif pathname == "/page-2":
        return display_salary()
    elif pathname == '/page-3':
        return display_participants()
    elif pathname == '/page-4':
        return display_dev_count()
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__ == '__main__':
    app.run_server(debug=True)
