import dash
import dash_core_components as dcc
import dash_html_components as html

from ebisim import SimpleEBISProblem

external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(className="container", children=[
    html.Section(className="jumbotron text-center", children=[
        html.H1(className="jumbotron-heading", children='EBISIM'),
        html.Div(children="\nA dashboard for simple ebisim tasks.\n")
    ]),
    html.Div(className="container", children=[
        "Z: ", dcc.Input(id="ctrl_z", value=20, type="number"),
        "j: ", dcc.Input(id="ctrl_j", value=200, type="number"), "A/cm**2",
        "E: ", dcc.Input(id="ctrl_e", value=3000, type="number"), "eV",
        "t: ", dcc.Input(id="ctrl_t", value=1, type="number"), "s",
    ]),
    dcc.Graph(id='plot_csevo')
])

@app.callback(
    dash.dependencies.Output("plot_csevo", 'figure'),
    [dash.dependencies.Input("ctrl_z", "value"), dash.dependencies.Input("ctrl_j", "value"),
     dash.dependencies.Input("ctrl_e", "value"), dash.dependencies.Input("ctrl_t", "value")]
)
def update_csevo(z, j, e_kin, tmax):
    sol = SimpleEBISProblem(int(z), j, e_kin, 1.).solve(tmax)
    data = []
    for cs in range(sol.y.shape[0]):
        data.append(
            {"x": sol.t, "y":sol.y[cs,:], "name":str(cs)+"+", "type":"line"}
        )
    layout = {
        "title":'Charge State Evolution',
        "xaxis":{"title":"Time (s)", "type":"log"},
        "yaxis":{"title":"Relative abundance"}
    }

    return {"data":data, "layout":layout}
if __name__ == "__main__":
    app.run_server(debug=True)