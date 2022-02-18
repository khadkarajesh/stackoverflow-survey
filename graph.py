def display_network_diagram():
    interest_df = survey_df.copy()
    interest_df.dropna(inplace=True)

    platform_df = interest_df.copy()

    platform_df["PlatformHaveWorkedWith"] = platform_df["PlatformHaveWorkedWith"].str.split(";")
    platform_df = platform_df.explode("PlatformHaveWorkedWith").reset_index(drop=True)
    platforms = platform_df[platform_df['PlatformHaveWorkedWith'].notnull()]['PlatformHaveWorkedWith'].unique().tolist()

    # print(platforms)

    nodes = []
    for index, platform in enumerate(platforms):
        nodes.append({'data': {'id': str(index + 1), 'label': platform}})

    connections = []
    for index, row in interest_df.iterrows():
        src_platforms = row['PlatformHaveWorkedWith'].split(";")
        dst_platforms = row['PlatformWantToWorkWith'].split(";")
        # print(src_platforms, dst_platforms, platforms.index(src_platforms[0]))
        for src_platform in src_platforms:
            for dst_platform in dst_platforms:
                connections.append({'data': {'source': f'{platforms.index(src_platform) + 1}',
                                             'target': f'{platforms.index(dst_platform) + 1}'}})

    # elements = list()
    # elements.append(nodes)
    nodes.append(connections)

    # print(nodes)

    # print(elements)

    return html.Div([
        cyto.Cytoscape(
            id='cytoscape',
            elements=nodes,
            layout={'name': 'circle'},
            # stylesheet=[{
            #     'selector': 'node',
            #     'style': {
            #         'height': 20,
            #         'width': 20,
            #         'background-color': '#e8e406'
            #     }
            # }, {
            #     'selector': 'edge',
            #     'style': {
            #         'curve-style': 'haystack',
            #         'haystack-radius': 0,
            #         'width': 5,
            #         'opacity': 0.5,
            #         'line-color': '#f2f08c'
            #     }
            # }],
            style={
                'width': '100%',
                'height': '100%',
                'position': 'absolute',
                'left': 0,
                'top': 0,
                'z-index': 999
            }
        )
    ])
