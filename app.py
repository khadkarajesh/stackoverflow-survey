import dash_bootstrap_components as dbc
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, html, dcc, Output, Input

survey_df = pd.read_csv("data.csv")

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
server = app.server
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
                dbc.NavLink("Salary of Developers", href="/page-1", active="exact"),
                dbc.NavLink("First line code written", href="/page-2", active="exact"),
                dbc.NavLink("Participants by Country", href="/page-3", active="exact"),
                dbc.NavLink("Participants by Developer", href="/page-4", active="exact"),
                dbc.NavLink("Education by Gender", href="/page-5", active="exact"),
                dbc.NavLink("Interested Cloud Platform", href="/page-6", active="exact"),
                dbc.NavLink("Gender vs Salary", href='/page-7', active="exact"),
                dbc.NavLink("Problem Resolution Method", href='/page-8', active="exact")
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
    elif x == '55 - 64 years':
        return 7
    return 8


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
                 color='Age1stCode',
                 orientation='h')

    fig.update_layout(xaxis_title="Percentage",
                      yaxis_title="Age Group",
                      title="First line of code written age percentage",
                      legend_title="Age Group")

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
    print(f"columns: {salary_exp_df.columns}")
    fig = px.scatter(salary_exp_df,
                     x="YearsCodePro",
                     y="ConvertedCompYearly",
                     color="Participants",
                     text='dev_type_mapper',
                     labels={
                         "YearsCodePro": "Average Years of Professional Experience",
                         "ConvertedCompYearly": "Median Yearly Salary(USD)",
                     }, custom_data=['DevType'])

    fig.update_traces(textposition='top center',
                      hovertemplate="<br>".join([
                          "Average Years of Experience: %{x}",
                          "Median Salary: %{y}",
                          "Developer Type: %{customdata[0]}"
                      ]))
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


def map_salary(x):
    if x < 10000:
        return "Low(<10,000)"
    elif 10000 <= x < 49000:
        return "Low Med(10k-49k)"
    elif 49000 <= x < 85000:
        return "Medium(49k-85k)"
    elif 85000 <= x < 150000:
        return "High(85k-150k)"
    return "Very High > 150k"


def map_education_label(x):
    ['Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)',
     'Bachelor’s degree (B.A., B.S., B.Eng., etc.)',
     'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)',
     'Other doctoral degree (Ph.D., Ed.D., etc.)',
     'Some college/university study without earning a degree',
     'Something else', 'Professional degree (JD, MD, etc.)',
     'Primary/elementary school', 'Associate degree (A.A., A.S., etc.)']
    if x == "Other doctoral degree":
        return x
    return x.split(" ")[0] + " " + x.split(" ")[1]


def set_salary_range_priority(x):
    if x == 'Low(<10,000)':
        return 0
    elif x == 'Low Med(10k-49k)':
        return 1
    elif x == 'Medium(49k-85k)':
        return 2
    elif x == 'High(85k-150k)':
        return 3
    elif x == 'Very High > 150k':
        return 4


def display_education_by_gender():
    degree_salary = survey_df.copy()
    degree_salary.dropna(inplace=True)
    degree_salary = degree_salary.groupby(['EdLevel']).agg(
        {'ConvertedCompYearly': np.mean, 'ResponseId': 'size'}).reset_index()
    degree_salary['salary_mapper'] = degree_salary['ConvertedCompYearly'].apply(map_salary)
    degree_salary['ed_mapper'] = degree_salary['EdLevel'].apply(map_education_label)
    degree_salary['salary_mapper_order'] = degree_salary['salary_mapper'].apply(set_salary_range_priority)
    degree_salary = degree_salary.sort_values(by=['salary_mapper_order'])

    fig = px.density_heatmap(degree_salary,
                             y='ed_mapper',
                             x='salary_mapper',
                             z='ResponseId')

    fig.update_layout(height=600,
                      title={
                          'text': "Salary By Education"},
                      xaxis={
                          'title': "Salary Range"
                      },
                      yaxis={
                          'title': 'Highest Level of Education'
                      },
                      coloraxis_colorbar=dict(
                          title="Number of Participants",
                      ))
    return html.Div([
        dcc.Graph(
            id='dev_type_counts',
            figure=fig
        )
    ])


def make_edge(x, y, text, width):
    return go.Scatter(x=x,
                      y=y,
                      line=dict(width=width,
                                color='cornflowerblue'),
                      hoverinfo='text',
                      text=([text]),
                      mode='lines')


def display_network_diagram():
    interest_df = survey_df.copy()
    interest_df.dropna(inplace=True)

    platform_df = interest_df.copy()

    platform_df = platform_df[['PlatformHaveWorkedWith', 'PlatformWantToWorkWith']]

    platform_df["PlatformHaveWorkedWith"] = platform_df["PlatformHaveWorkedWith"].str.split(";")
    platform_df = platform_df.explode("PlatformHaveWorkedWith").reset_index(drop=True)

    platform_df["PlatformWantToWorkWith"] = platform_df["PlatformWantToWorkWith"].str.split(";")
    platform_df = platform_df.explode("PlatformWantToWorkWith").reset_index(drop=True)

    G = nx.from_pandas_edgelist(platform_df, 'PlatformHaveWorkedWith', 'PlatformWantToWorkWith')

    pos = nx.circular_layout(G)

    for n, p in pos.items():
        G.nodes[n]['pos'] = p

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=list(pos.keys()),
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='RdBu',
            reversescale=True,
            color=[],
            size=15,
            colorbar=dict(
                thickness=10,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=0)))

    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

    for node, adjacencies in enumerate(G.adjacency()):
        node_trace['marker']['color'] += tuple([len(adjacencies[1])])
        node_info = adjacencies[0] + ' # of connections: ' + str(len(adjacencies[1]))
        node_trace['text'] += tuple([node_info])

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Cloud Platform worked and wanted to work in future',
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        # annotations=[dict(
                        #     text="No. of connections",
                        #     showarrow=False,
                        #     xref="paper", yref="paper")],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    return html.Div([
        dcc.Graph(
            id='dev_type_counts',
            figure=fig
        )
    ])


def map_gender(x):
    return x if x in ['Man', 'Woman'] else "Prefer not to say"


def map_experience(x):
    if x == 'Less than 1 year':
        return 1
    elif x == 'More than 50 years':
        return 50
    return x


def display_salary_by_gender():
    salary_by_gender = survey_df.copy()
    salary_by_gender['Gender'] = salary_by_gender['Gender'].apply(map_gender)
    salary_by_gender['YearsCodePro'] = salary_by_gender['YearsCodePro'].apply(map_experience)
    salary_by_gender['YearsCodePro'] = salary_by_gender['YearsCodePro'].fillna("0")
    salary_by_gender['YearsCodePro'] = salary_by_gender['YearsCodePro'].astype('int')
    salary_by_gender = salary_by_gender[
        (salary_by_gender['YearsCodePro'] <= 30) & (salary_by_gender['YearsCodePro'] > 1)]
    # salary_by_gender = salary_by_gender[salary_by_gender['ConvertedCompYearly'] <= 300000]
    salary_by_gender = salary_by_gender[salary_by_gender['Gender'] != 'Prefer not to say']
    # salary_by_gender['ConvertedCompYearly'].fillna(salary_by_gender['ConvertedCompYearly'].mean())
    # salary_by_gender['YearsCodePro'].fillna(salary_by_gender['YearsCodePro'].mean())
    salary_by_gender = salary_by_gender.interpolate(method="akima")
    # salary_by_gender.dropna(inplace=True)
    new_df = salary_by_gender.groupby(['Gender', 'YearsCodePro']).agg(
        {'YearsCodePro': 'mean', 'ConvertedCompYearly': 'median'})
    new_df['Gender'] = new_df.index.get_level_values(0)
    fig = px.line(new_df, x="YearsCodePro", y="ConvertedCompYearly", color='Gender')

    fig.update_layout(title={'text': 'Salary Comparison by Gender'},
                      xaxis_title="Average Years of Professional Experience",
                      yaxis_title="Median Yearly Salary(USD)",
                      legend_title="Gender")

    return html.Div([
        dcc.Graph(
            id='salary_by_gender',
            figure=fig
        )
    ])


def display_home_screen():
    dashboard_df = survey_df.copy()

    learned_code_df = dashboard_df.copy()
    learned_code_df["LearnCode"] = learned_code_df["LearnCode"].str.split(";")

    learned_code_df = learned_code_df.explode("LearnCode").reset_index(drop=True)
    learned_code_df = learned_code_df.groupby('LearnCode').size().reset_index(name='size')

    learned_code_fig = px.histogram(learned_code_df, x='LearnCode', y='size', color='LearnCode')
    learned_code_fig.update_layout(
        xaxis={'categoryorder': 'total descending', 'showticklabels': False, 'title': 'Learned Code'},
        yaxis={'title': 'Number of Participants'},
        legend_title='Source')

    popular_programming_df = dashboard_df.copy()
    popular_programming_df["LanguageHaveWorkedWith"] = popular_programming_df["LanguageHaveWorkedWith"].str.split(";")
    popular_programming_df = popular_programming_df.explode("LanguageHaveWorkedWith").reset_index(drop=True)
    popular_programming_df = popular_programming_df.groupby('LanguageHaveWorkedWith').size().reset_index(name='size')

    popular_programming_df = popular_programming_df.sort_values(by=['size'], ascending=False)
    # popular_programming_df = popular_programming_df.head(10)

    popular_programming_fig = px.bar(popular_programming_df, y='LanguageHaveWorkedWith', x='size', orientation='h')
    popular_programming_fig.update_layout(xaxis={'title': 'Number of Participants worked'},
                                          yaxis={'title': '',
                                                 'categoryorder': 'total ascending'},
                                          legend_title='Source')

    popular_database_df = dashboard_df.copy()

    popular_database_df["DatabaseHaveWorkedWith"] = popular_database_df["DatabaseHaveWorkedWith"].str.split(";")
    popular_database_df = popular_database_df.explode("DatabaseHaveWorkedWith").reset_index(drop=True)
    popular_database_df = popular_database_df.groupby('DatabaseHaveWorkedWith').size().reset_index(name='size')

    popular_database_df = popular_database_df.sort_values(by=['size'], ascending=False)
    # popular_database_df = popular_database_df.head(10)

    popular_database_fig = px.bar(popular_database_df, y='DatabaseHaveWorkedWith', x='size', orientation='h')
    popular_database_fig.update_layout(width=1000, xaxis={'title': 'Number of Participants worked'},
                                       yaxis={'title': '',
                                              'categoryorder': 'total ascending'},
                                       legend_title='Source')

    popular_framework_df = dashboard_df.copy()
    popular_framework_df["WebframeHaveWorkedWith"] = popular_framework_df["WebframeHaveWorkedWith"].str.split(";")
    popular_framework_df = popular_framework_df.explode("WebframeHaveWorkedWith").reset_index(drop=True)
    popular_framework_df = popular_framework_df.groupby('WebframeHaveWorkedWith').size().reset_index(name='size')

    popular_framework_df = popular_framework_df.sort_values(by=['size'], ascending=False)
    # popular_framework_df = popular_framework_df.head(10)

    popular_framework_fig = px.bar(popular_framework_df, y='WebframeHaveWorkedWith', x='size', orientation='h')
    popular_framework_fig.update_layout(width=1000, xaxis={'title': 'Number of Participants worked'},
                                        yaxis={'title': '',
                                               'categoryorder': 'total ascending'},
                                        legend_title='Source')

    popular_tools_df = dashboard_df.copy()
    popular_tools_df["ToolsTechHaveWorkedWith"] = popular_tools_df["ToolsTechHaveWorkedWith"].str.split(";")
    popular_tools_df = popular_tools_df.explode("ToolsTechHaveWorkedWith").reset_index(drop=True)
    popular_tools_df = popular_tools_df.groupby('ToolsTechHaveWorkedWith').size().reset_index(name='size')

    popular_tools_df = popular_tools_df.sort_values(by=['size'], ascending=False)
    # popular_tools_df = popular_tools_df.head(10)

    popular_tools_fig = px.bar(popular_tools_df, y='ToolsTechHaveWorkedWith', x='size', orientation='h')
    popular_tools_fig.update_layout(width=1000, xaxis={'title': 'Number of Participants worked'},
                                    yaxis={'title': '',
                                           'categoryorder': 'total ascending'},
                                    legend_title='Source')

    popular_ide_df = dashboard_df.copy()
    popular_ide_df["NEWCollabToolsHaveWorkedWith"] = popular_ide_df["NEWCollabToolsHaveWorkedWith"].str.split(";")
    popular_ide_df = popular_ide_df.explode("NEWCollabToolsHaveWorkedWith").reset_index(drop=True)
    popular_ide_df = popular_ide_df.groupby('NEWCollabToolsHaveWorkedWith').size().reset_index(name='size')

    popular_ide_df = popular_ide_df.sort_values(by=['size'], ascending=False)
    # popular_tools_df = popular_tools_df.head(10)

    popular_ide_fig = px.bar(popular_ide_df, y='NEWCollabToolsHaveWorkedWith', x='size', orientation='h')
    popular_ide_fig.update_layout(width=1000, xaxis={'title': 'Number of Participants'},
                                  yaxis={'title': '',
                                         'categoryorder': 'total ascending'},
                                  legend_title='Source')

    return html.Div([
        html.Div([
            html.H5("Learning How to Code"),
            dcc.Graph(
                id='learned_code_fig',
                figure=learned_code_fig
            )
        ], className='row'),
        html.Div([
            html.H5("Programming, Scripting and Markup Languages"),
            dcc.Graph(
                id='popular_programming_fig',
                figure=popular_programming_fig
            )
        ], className='column'),
        html.Div([
            html.H5("Databases"),
            dcc.Graph(
                id='popular_database_fig',
                figure=popular_database_fig
            )
        ], className='row'),
        html.Div([
            html.H5("Frameworks"),
            dcc.Graph(
                id='popular_framework_fig',
                figure=popular_framework_fig
            )
        ], className='row'),
        html.Div([
            html.H5("Used Tools and Technologies"),
            dcc.Graph(
                id='popular_tools_fig',
                figure=popular_tools_fig
            )
        ], className='row'),
        html.Div([
            html.H5("IDE used for Development"),
            dcc.Graph(
                id='popular_tools_fig',
                figure=popular_ide_fig
            )
        ], className='row')
    ], className='column')


def display_resolution_method():
    resolution_df = survey_df.copy()

    resolution_df["NEWStuck"] = resolution_df["NEWStuck"].str.split(";")
    resolution_df = resolution_df.explode("NEWStuck").reset_index(drop=True)

    resolution_df_man = resolution_df[resolution_df['Gender'] == 'Man']
    resolution_df_women = resolution_df[resolution_df['Gender'] == 'Woman']

    resolution_df_man = resolution_df_man.groupby('NEWStuck').size().reset_index(name='size')
    resolution_df_woman = resolution_df_women.groupby('NEWStuck').size().reset_index(name='size')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=resolution_df_man['NEWStuck'].values,
        x=resolution_df_man['size'].values,
        name='Man',
        orientation='h',
        marker=dict(
            color='rgba(58, 71, 80, 0.6)',
            line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
        )
    ))
    fig.add_trace(go.Bar(
        y=resolution_df_woman['NEWStuck'].values,
        x=resolution_df_woman['size'].values,
        name='Women',
        orientation='h',

        marker=dict(
            color='rgba(246, 78, 139, 0.6)',
            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
        )
    ))

    fig.update_layout(title={'text': "Followed approach when stuck on some problem"},
                      barmode='stack',
                      yaxis={'categoryorder': 'total ascending'},
                      xaxis={'title': 'Number of Participants'})

    return html.Div([
        dcc.Graph(
            id='when_you_stuck',
            figure=fig
        )
    ])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return display_home_screen()
    elif pathname == "/page-1":
        return display_salary()
    elif pathname == "/page-2":
        return get_first_code_age()
    elif pathname == '/page-3':
        return display_participants()
    elif pathname == '/page-4':
        return display_dev_count()
    elif pathname == "/page-5":
        return display_education_by_gender()
    elif pathname == "/page-6":
        return display_network_diagram()
    elif pathname == '/page-7':
        return display_salary_by_gender()
    elif pathname == '/page-8':
        return display_resolution_method()
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__ == '__main__':
    app.run_server(debug=True)
