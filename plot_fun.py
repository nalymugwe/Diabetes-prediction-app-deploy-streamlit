import plotly.express as px 


def plot_model_results(df,metrics):
    title = f'Model Comparison {metrics}'
    
    color_discrete_map={
        'f1_score': 'blue',
        'accuracy_score':'grey'
    }
    fig  = px.bar(df, 
                x='model_name',
                y=[str(metrics)], barmode='group', 
                labels={'value': 'Score', 'model_name': 'Model'},
                color_discrete_map=color_discrete_map
                        )
    
    fig.update_layout(title=title,showlegend=False)
    return fig