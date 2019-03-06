"""
A small dashboard for very simple ebisim simulations
"""

import math

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

import ebisim

_BOOTSTRAP_CDN = "https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"
_DISCLAIMER = 'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.'

_ELEMENTS = [{"label":ebisim.elements.element_name(z), "value":z} for z in range(2, 106)]

_HEADER = html.Section(className="text-center", children=[
    html.H1("ebisim dash.", className="mb-3"),
    html.P("\nA dashboard for simple ebisim tasks.\n", className="lead text-muted"),
    html.Hr()
])

_FOOTER = html.Section(className="text-justify", children=[
    html.Hr(),
    html.H4('DISCLAIMER', className="text-muted mb-3"),
    html.P(_DISCLAIMER, className="text-muted")
])

_CONTROLS = html.Div(className="container", children=[
    html.P("Confirm entries by pressing enter or switching focus."),
    html.Div(className="row", children=[
        html.Div(className="col-md mb-3", children=[
            html.Label("Element", htmlFor="ctrl_element"),
            dcc.Dropdown(id="ctrl_element", options=_ELEMENTS, value=20)
        ]),
        html.Div(className="col-md mb-3", children=[
            html.Label("Breeding time (ms)", htmlFor="ctrl_brtime"),
            dcc.Input(id="ctrl_brtime", value=200, min=1, type="number", className="form-control",
                      debounce=True)
        ])
    ]),
    html.Div(className="row", children=[
        html.Div(className="col-md mb-3", children=[
            html.Label("Current density (A/cm^2)", htmlFor="ctrl_curden"),
            dcc.Input(id="ctrl_curden", value=100, min=1, type="number", className="form-control",
                      debounce=True)
        ]),
        html.Div(className="col-md mb-3", children=[
            html.Label("Beam energy (eV)", htmlFor="ctrl_energy"),
            dcc.Input(id="ctrl_energy", value=5000, min=1, type="number", className="form-control",
                      debounce=True)
        ]),
        html.Div(className="col-md mb-3", children=[
            html.Label("Beam energy FWHM (eV)", htmlFor="ctrl_fwhm"),
            dcc.Input(id="ctrl_fwhm", value=50, min=1, type="number", className="form-control",
                      debounce=True)
        ])
    ])
])


app = dash.Dash(__name__, external_stylesheets=[_BOOTSTRAP_CDN])

app.layout = html.Div(className="container", children=[
    _HEADER,
    _CONTROLS,
    dcc.Graph(id='plot_csevo', style={'height': 700},),
    _FOOTER
])

@app.callback(Output("plot_csevo", 'figure'), [
    Input("ctrl_element", "value"), Input("ctrl_curden", "value"), Input("ctrl_energy", "value"),
    Input("ctrl_fwhm", "value"), Input("ctrl_brtime", "value")
])
def update_csevo(z, j, e_kin, fwhm, tmax):
    """This function creates the charge state evolution plot"""
    tmax /= 1000

    try:
        sol = ebisim.SimpleEBISProblem(int(z), j, e_kin, fwhm).solve(tmax)
    except:
        sol = None

    data = []
    if sol:
        for cs in range(sol.y.shape[0]):
            data.append(
                {"x": sol.t, "y":sol.y[cs, :], "name":str(cs)+"+", "type":"line"}
            )

    highlim = math.log10(tmax)
    lowlim = math.floor(math.log10(.01/j))
    lowlim = lowlim if (lowlim < highlim) else highlim-1
    layout = {
        "title":'Charge State Evolution',
        "template":"plotly_dark",
        "xaxis":{"title":"Time (s)", "type":"log", "range":[lowlim, highlim]},
        "yaxis":{"title":"Relative abundance"}
    }

    return {"data":data, "layout":layout}

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0")
