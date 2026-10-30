import subprocess
import sys

try:
    import matplotlib as mpl
    import matplotlib.pyplot
except ImportError:
    pass

import numpy as np
import pytest

import skrf as rf
from skrf.networkSet import NetworkSet

if "matplotlib" not in sys.modules:
    pytest.skip(allow_module_level=True)


@pytest.fixture(params=rf.Network.PRIMARY_PROPERTIES)
def primary_properties(request):
    return request.param

@pytest.fixture(params=["polar", "complex"])
def primary_methods(request):
    return request.param

@pytest.fixture(params=rf.Network._generated_functions().keys())
def generated_functions(request):
    return request.param

def test_primary_plotting(ntwk1_dc: rf.Network, primary_properties: str, primary_methods: str):
    method = f"plot_{primary_properties}_{primary_methods}"
    fig = getattr(ntwk1_dc, method)()

def test_generated_function_plots(ntwk1_dc: rf.Network, generated_functions: str):
    method = f"plot_{generated_functions}"
    fig = getattr(ntwk1_dc, method)()

def test_plot_passivity(ntwk1_dc: rf.Network):
    ntwk1_dc.plot_passivity()

def test_plot_reciprocity(ntwk1_dc: rf.Network, ):
    ntwk1_dc.plot_reciprocity()

def test_plot_reciprocity2(ntwk1_dc: rf.Network, ):
    ntwk1_dc.plot_reciprocity2()

def test_plot_s_db_time(ntwk1_dc: rf.Network, ):
    ntwk1_dc.plot_s_db_time()
    ntwk1_dc.plot_s_db_time(m=0, n=0)

def test_plot_s_error(ntwk1_dc: rf.Network):
    ntwk1_dc.plot_s_error(ntwk1_dc)
    ntwk1_dc.plot_s_error(ntwk1_dc, error_function='average_l1_norm')
    ntwk1_dc.plot_s_error(ntwk1_dc, db=True)
    rf.plotting.plot_s_error(ntwk1_dc, ntwk1_dc, error_function='maximum_l1_norm')

def test_plot_s_smith(ntwk1_dc: rf.Network, ):
    ntwk1_dc.plot_s_smith()

def test_z_time_impulse_step_z0(ntwk1_dc: rf.Network, ):
    se_diff = rf.network.concat_ports([ntwk1_dc, ntwk1_dc], port_order = 'second')
    mm_diff = se_diff.copy()
    mm_diff.se2gmm(p = 2)
    fig, ax = mpl.pyplot.subplots(1, 1)
    mm_diff.plot_z_time_step(0, 0, ax = ax)
    mm_diff.plot_z_time_impulse(0, 1, ax = ax)
    mm_diff.plot_z_time_step(2, 2, ax = ax)
    mm_diff.plot_z_time_impulse(2, 3, ax = ax)
    # test that start impedance is almost port z0 to 0.1%
    z0_start = [
        ax.lines[0].get_ydata()[0],
        ax.lines[1].get_ydata()[0],
        ax.lines[2].get_ydata()[0],
        ax.lines[3].get_ydata()[0]]
    np.testing.assert_allclose(z0_start, mm_diff.z0[0],
        rtol = 1e-3,
        err_msg = "plot_z_time_xxx start impedance does not match port impedance")

def test_x_y_labels(ntwk1_dc: rf.Network):
    fig, ax = mpl.pyplot.subplots(1, 1)
    ntwk1_dc.plot_s_db(ax = ax)
    # check that there are 4 traces
    np.testing.assert_equal(len(ax.lines), ntwk1_dc.nports**2,
                                    err_msg = "missing trace")
    # check x label
    np.testing.assert_string_equal(ax.get_xlabel(), f"Frequency ({ntwk1_dc.frequency.unit})")
    # check y label
    np.testing.assert_string_equal(ax.get_ylabel(), rf.Network.Y_LABEL_DICT['db'])



@pytest.mark.parametrize("usetex", [True, False])
def test_plot_it_all(ntwk1_dc: rf.Network, usetex):
    with mpl.rc_context({'text.usetex': usetex}):
        ntwk1_dc.plot_it_all()

@pytest.mark.parametrize("usetex", [True, False])
def test_plot_polar(ntwk1_dc: rf.Network, usetex):
    with mpl.rc_context({'text.usetex': usetex}):
        ntwk1_dc.plot_s_polar()

def test_plot_uncertainty_bounds_s(ntwk_set_zip: NetworkSet):
    ntwk_set_zip.plot_uncertainty_bounds_s()

def test_plot_uncertainty_bounds_s_db(ntwk_set_zip: NetworkSet):
    ntwk_set_zip.plot_uncertainty_bounds_s_db()

def test_plot_uncertainty_bounds_s_time_db(ntwk_set_zip: NetworkSet):
    ntwk_set_zip.plot_uncertainty_bounds_s_time_db()

def test_plot_uncertainty_decomposition(ntwk_set_zip: NetworkSet):
    ntwk_set_zip.plot_uncertainty_decomposition()

def test_plot_uncertainty_decomposition_title(ntwk1: rf.Network):
    NetworkSet([ntwk1, ntwk1], name="set").plot_uncertainty_decomposition()
    assert mpl.pyplot.gca().get_title() == "Uncertainty Decomposition: set $S_{11}$"

def test_plot_minmax_bounds_s_db(ntwk_set_zip: NetworkSet):
    ntwk_set_zip.plot_minmax_bounds_s_db()

def test_plot_minmax_bounds_s_db10(ntwk_set_zip: NetworkSet):
    ntwk_set_zip.plot_minmax_bounds_s_db10()

def test_plot_minmax_bounds_s_time_db(ntwk_set_zip: NetworkSet):
    ntwk_set_zip.plot_minmax_bounds_s_time_db()

def test_plot_signature(ntwk_set_zip: NetworkSet):
    ntwk_set_zip.signature()

def test_generated_violin_plots(ntwk_set_zip: NetworkSet, generated_functions: str):
    if "time" not in generated_functions:
        fig = ntwk_set_zip.plot_violin(generated_functions)
    else:
        with pytest.raises(NotImplementedError):
            fig = ntwk_set_zip.plot_violin(generated_functions)


# Plotting backend selection (skrf.plotting.options.backend) and the plotly backend

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

needs_plotly = pytest.mark.skipif(go is None, reason="plotly not installed")


@pytest.fixture(autouse=True)
def reset_plotting_backend():
    yield
    rf.plotting.options.backend = "matplotlib"

@pytest.fixture
def mpl_axes():
    fig, ax = mpl.pyplot.subplots(1, 1)
    yield ax
    mpl.pyplot.close("all")

def _modules_after(code: str) -> set:
    out = subprocess.run([sys.executable, "-c", f"{code}; import sys; print(' '.join(sys.modules))"],
                         capture_output=True, text=True, check=True)
    return {m.split(".")[0] for m in out.stdout.split()}

def assert_plotly_parity(fig, ax, x_scale: float = 1.0):
    """Plotly traces carry the same data, names and y label as the matplotlib lines on `ax`."""
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == len(ax.lines)
    for trace, line in zip(fig.data, ax.lines):
        np.testing.assert_allclose(np.asarray(trace.x, dtype=float) * x_scale, line.get_xdata(), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(trace.y, dtype=float), line.get_ydata(), rtol=1e-12)
        if not line.get_label().startswith("_"):
            assert trace.name == line.get_label()
    assert fig.layout.yaxis.title.text == (ax.get_ylabel() or None)

def test_backend_option():
    assert rf.plotting.options.backend == "matplotlib"
    rf.plotting.options.backend = "plotly"
    assert rf.plotting.options.backend == "plotly"
    with pytest.raises(ValueError, match="matplotlib.*plotly"):
        rf.plotting.options.backend = "bokeh"
    assert rf.plotting.options.backend == "plotly"

def test_plotly_missing_raises_import_error(ntwk1: rf.Network, monkeypatch):
    for mod in [m for m in sys.modules if m.split(".")[0] == "plotly"]:
        monkeypatch.delitem(sys.modules, mod)
    monkeypatch.setitem(sys.modules, "plotly", None)
    with pytest.raises(ImportError, match=r"scikit-rf\[plotly\]"):
        ntwk1.plot_s_db(backend="plotly")

@needs_plotly
def test_plotly_not_implemented_does_not_fall_back(ntwk_set_zip: NetworkSet):
    rf.plotting.options.backend = "plotly"
    with pytest.raises(NotImplementedError, match="plotly"):
        ntwk_set_zip.plot_violin("s_db", quantiles=[0.5])

def test_matplotlib_unchanged_under_global_plotly(ntwk1: rf.Network, ntwk_set_zip: NetworkSet, mpl_axes):
    rf.plotting.options.backend = "plotly"
    # per-call override keeps the matplotlib return values
    assert ntwk1.plot_s_db(backend="matplotlib") is None
    assert ntwk1.frequency.plot(ntwk1.s_db[:, 0, 0], backend="matplotlib") is None
    assert ntwk1.plot_passivity(backend="matplotlib") is None
    assert ntwk_set_zip.plot_s_db(backend="matplotlib") == [None] * len(ntwk_set_zip)
    # a matplotlib ax selects matplotlib, so code drawing on its own Axes keeps working
    assert ntwk1.plot_s_db(0, 0, ax=mpl_axes) is None
    assert ntwk1.plot_s_smith(ax=mpl_axes) is None
    assert ntwk1.plot_s_complex(0, 0, mpl_axes) is None
    assert ntwk_set_zip.plot_uncertainty_bounds_s_db(ax=mpl_axes) is None
    assert ntwk_set_zip.plot_violin("s_db", ax=mpl_axes) is None
    assert len(mpl_axes.lines) > 0
    # matplotlib-only helpers ignore the global option
    n_lines = len(mpl_axes.lines)
    rf.media.Freespace(ntwk1.frequency).plot_ep()
    assert len(mpl_axes.lines) == n_lines + 2

def test_plot_prop_keyword_args(ntwk1: rf.Network, mpl_axes):
    ntwk1.plot_prop_complex(prop_name="s", m=0, n=0, ax=mpl_axes)
    assert len(mpl_axes.lines) == 1

@needs_plotly
def test_plotly_does_not_import_matplotlib():
    loaded = _modules_after(
        "import skrf as rf; rf.plotting.options.backend = 'plotly'; n = rf.data.ring_slot; "
        "n.plot_s_db(); n.plot_s_time_db(); n.frequency.plot(n.s_db); n.plot_passivity(); "
        "n.plot_s_smith(); n.plot_s_polar(); n.plot_s_complex(); n.plot_it_all(); "
        "ns = rf.networkSet.NetworkSet([n, n]); ns.plot_uncertainty_bounds_s_db(); ns.plot_minmax_bounds_s_db(); "
        "ns.plot_violin('s_db'); ns.plot_uncertainty_decomposition(); s = n.s[:, 0, 0]; "
        "rf.plotting.plot_rectangular(n.f, n.s_db[:, 0, 0]); rf.plotting.plot_complex_rectangular(s); "
        "rf.plotting.plot_complex_polar(s); rf.plotting.plot_smith(s); rf.plotting.subplot_params(n)")
    assert "plotly" in loaded
    assert "matplotlib" not in loaded

@needs_plotly
def test_plotly_generated_function_parity(ntwk1_dc: rf.Network, generated_functions: str, mpl_axes):
    method = f"plot_{generated_functions}"
    fig = getattr(ntwk1_dc, method)(backend="plotly")
    getattr(ntwk1_dc, method)(ax=mpl_axes)
    is_freq = "time" not in generated_functions
    assert len(fig.data) == ntwk1_dc.nports**2
    assert_plotly_parity(fig, mpl_axes, x_scale=ntwk1_dc.frequency.multiplier if is_freq else 1.0)
    assert fig.layout.xaxis.title.text == (f"Frequency ({ntwk1_dc.frequency.unit})" if is_freq else "Time (ns)")

@needs_plotly
@pytest.mark.parametrize("call", [
    lambda n, ns, **kw: n.plot_passivity(**kw),
    lambda n, ns, **kw: n.plot_reciprocity2(**kw),
    lambda n, ns, **kw: n.plot_s_error(n, **kw),
    lambda n, ns, **kw: n.frequency.plot(n.s_db, **kw),
    lambda n, ns, **kw: ns.plot_s_db(**kw),
    lambda n, ns, **kw: ns.plot_logsigma(**kw),
    lambda n, ns, **kw: ns.plot_uncertainty_decomposition(**kw),
], ids=["plot_passivity", "plot_reciprocity2", "plot_s_error", "Frequency.plot", "NetworkSet.plot_s_db",
        "plot_logsigma", "plot_uncertainty_decomposition"])
def test_plotly_rectangular_parity(ntwk1_dc: rf.Network, ntwk_set_zip: NetworkSet, call, mpl_axes):
    fig = call(ntwk1_dc, ntwk_set_zip, backend="plotly")
    call(ntwk1_dc, ntwk_set_zip, ax=mpl_axes)
    assert_plotly_parity(fig, mpl_axes, x_scale=ntwk1_dc.frequency.multiplier)

@needs_plotly
def test_plotly_figure_overlay_and_units(ntwk1: rf.Network):
    fig = go.Figure()
    assert ntwk1.plot_s_db(0, 0, ax=fig) is fig  # a plotly ax selects plotly
    mhz = ntwk1.copy()
    mhz.frequency.unit = "MHz"
    assert mhz.plot_s_db(0, 0, ax=fig) is fig
    assert len(fig.data) == 2
    np.testing.assert_allclose(fig.data[1].x, fig.data[0].x)  # rescaled to the figure's first unit
    assert fig.layout.xaxis.title.text == f"Frequency ({ntwk1.frequency.unit})"

@needs_plotly
def test_plotly_logx(ntwk1: rf.Network, mpl_axes):
    assert ntwk1.plot_s_db(0, 0, logx=True, backend="plotly").layout.xaxis.type == "log"
    ntwk1.plot_s_db(0, 0, logx=True, ax=mpl_axes)
    assert mpl_axes.get_xscale() == "log"

@needs_plotly
def test_plotly_errors(ntwk1: rf.Network, mpl_axes):
    with pytest.raises(TypeError, match="Positional"):
        ntwk1.frequency.plot(ntwk1.s_db[:, 0, 0], "r--", backend="plotly")
    with pytest.raises(TypeError, match="cannot be used with backend"):
        ntwk1.plot_s_db(ax=mpl_axes, backend="plotly")

@needs_plotly
def test_plotly_kwargs_translation(ntwk1: rf.Network):
    trace = ntwk1.plot_s_db(0, 0, c="r", ls="--", lw=2, marker="o", ms=4, alpha=0.5, label="a",
                            backend="plotly").data[0]
    assert (trace.line.color, trace.line.dash, trace.line.width) == ("red", "dash", 2)
    assert (trace.marker.symbol, trace.marker.size, trace.mode) == ("circle", 4, "lines+markers")
    assert (trace.opacity, trace.name) == (0.5, "a")
    trace = ntwk1.plot_s_db(0, 0, marker=dict(size=5), backend="plotly").data[0]
    assert (trace.marker.size, trace.mode) == (5, "lines")

def _smith_grid_traces(fig) -> list:
    return [t for t in fig.data if t.meta == "skrf_smith_chart"]

@needs_plotly
def test_plotly_complex_parity(ntwk1_dc: rf.Network, primary_properties: str, mpl_axes):
    method = f"plot_{primary_properties}_complex"
    fig = getattr(ntwk1_dc, method)(backend="plotly")
    getattr(ntwk1_dc, method)(ax=mpl_axes)
    assert len(fig.data) == ntwk1_dc.nports**2
    assert_plotly_parity(fig, mpl_axes)
    assert (fig.layout.xaxis.title.text, fig.layout.yaxis.scaleanchor) == ("Real", "x")
    assert (fig.layout.xaxis.constrain, fig.layout.yaxis.constrain) == ("domain", "domain")
    assert (fig.layout.width, fig.layout.height) == (700, 500)
    z = getattr(ntwk1_dc, primary_properties)
    np.testing.assert_allclose(fig.layout.yaxis.range, [z.imag.min() - 0.05 * np.ptp(z.imag),
                                                        z.imag.max() + 0.05 * np.ptp(z.imag)])

@needs_plotly
def test_plotly_polar(ntwk1_dc: rf.Network, primary_properties: str):
    fig = getattr(ntwk1_dc, f"plot_{primary_properties}_polar")(backend="plotly")
    z = getattr(ntwk1_dc, primary_properties)
    ports = [(m, n) for m in range(ntwk1_dc.nports) for n in range(ntwk1_dc.nports)]
    assert len(fig.data) == len(ports)
    for trace, (m, n) in zip(fig.data, ports):
        assert (trace.type, trace.thetaunit) == ("scatterpolar", "radians")
        np.testing.assert_allclose(trace.r, np.abs(z[:, m, n]))
        np.testing.assert_allclose(trace.theta, np.angle(z[:, m, n]))
        assert trace.name == rf.plotting._get_label_str(ntwk1_dc, primary_properties.upper(), m, n, usetex=False)

@needs_plotly
@pytest.mark.parametrize("chart_type", ["z", "y", "zy", "yz"])
def test_plotly_smith(ntwk1_dc: rf.Network, chart_type: str):
    r = 0.5
    fig = ntwk1_dc.plot_s_smith(r=r, chart_type=chart_type, label_axes=True, backend="plotly")
    grid, data = _smith_grid_traces(fig), [t for t in fig.data if t.meta != "skrf_smith_chart"]
    assert len(data) == ntwk1_dc.nports**2
    np.testing.assert_allclose(data[1].x, ntwk1_dc.s[:, 0, 1].real)
    np.testing.assert_allclose(data[1].y, ntwk1_dc.s[:, 0, 1].imag)
    grid_points = np.concatenate([np.asarray(t.x, dtype=float) + 1j * np.asarray(t.y, dtype=float) for t in grid])
    assert np.nanmax(np.abs(grid_points)) <= r * (1 + 1e-9)
    assert ("whitesmoke" in [t.line.color for t in grid]) == (chart_type in ("zy", "yz"))
    np.testing.assert_allclose(fig.layout.xaxis.range, [-1.1 * r, 1.1 * r])
    assert (fig.layout.xaxis.constrain, fig.layout.yaxis.constrain) == ("domain", "domain")
    assert (fig.layout.width, fig.layout.height) == (700, 550)
    assert ntwk1_dc.plot_s_smith(ax=go.Figure(layout_height=300), backend="plotly").layout.width is None
    assert (fig.layout.xaxis.title.text, fig.layout.yaxis.title.text) == ("Real", "Imaginary")

@needs_plotly
def test_plotly_smith_labels(ntwk1_dc: rf.Network):
    fig = ntwk1_dc.plot_s_smith(draw_labels=True, draw_vswr=[2, 5], backend="plotly")
    texts = [a.text for a in fig.layout.annotations]
    assert {"∞", "0.0", "1.0", "1.0j", "2", "5"} <= set(texts)
    assert ntwk1_dc.plot_s_smith(draw_labels=True, chart_type="y", backend="plotly").layout.annotations[0].text \
        == "5.0"  # admittance labels, as matplotlib

@needs_plotly
def test_plotly_smith_grid(mpl_axes):
    fig = rf.plotting.smith(draw_labels=True, ref_imm=50.0, backend="plotly")
    rf.plotting.smith(draw_labels=True, ref_imm=50.0, ax=mpl_axes)
    assert len(_smith_grid_traces(fig)) > 0 and not [t for t in fig.data if t.meta != "skrf_smith_chart"]
    assert [a.text for a in fig.layout.annotations] == [t.get_text().replace(r"$\infty$", "∞") for t in mpl_axes.texts]

@needs_plotly
@pytest.mark.parametrize("kind", ["complex", "polar", "smith"])
def test_plotly_geometric_overlay(ntwk1: rf.Network, kind: str):
    fig = getattr(ntwk1, f"plot_s_{kind}")(0, 0, backend="plotly")
    n_traces, n_grid = len(fig.data), len(_smith_grid_traces(fig))
    assert getattr(ntwk1, f"plot_s_{kind}")(1, 1, ax=fig) is fig
    assert len(fig.data) == n_traces + 1
    assert len(_smith_grid_traces(fig)) == n_grid  # the smith chart is drawn once

@needs_plotly
def test_plotly_networkset_smith(ntwk1: rf.Network, ntwk_set_zip: NetworkSet):
    fig = ntwk_set_zip.plot_s_smith(backend="plotly")
    assert len(fig.data) - len(_smith_grid_traces(fig)) == sum(n.nports**2 for n in ntwk_set_zip)
    assert len(_smith_grid_traces(fig)) == len(_smith_grid_traces(ntwk1.plot_s_smith(backend="plotly")))

@needs_plotly
@pytest.mark.parametrize("method", ["plot_uncertainty_bounds_s_db", "plot_minmax_bounds_s_db",
                                    "plot_uncertainty_bounds_s"])
def test_plotly_bounds_parity(ntwk_set_zip: NetworkSet, method: str, mpl_axes):
    fig = getattr(ntwk_set_zip, method)(backend="plotly")
    getattr(ntwk_set_zip, method)(ax=mpl_axes)
    means, fills = fig.data[::2], fig.data[1::2]
    assert len(means) == len(mpl_axes.lines) == len(mpl_axes.collections)
    for mean, fill, line, poly in zip(means, fills, mpl_axes.lines, mpl_axes.collections):
        np.testing.assert_allclose(np.asarray(mean.x) * ntwk_set_zip[0].frequency.multiplier, line.get_xdata())
        np.testing.assert_allclose(mean.y, line.get_ydata())
        assert mean.name == line.get_label()
        np.testing.assert_allclose(np.unique(fill.y), np.unique(poly.get_paths()[0].vertices[:, 1]))
        assert (fill.fill, fill.fillcolor, fill.showlegend) == ("toself", mean.line.color, False)
    assert fig.layout.yaxis.title.text == mpl_axes.get_ylabel()
    assert getattr(ntwk_set_zip, method)(ax=fig) is fig
    assert len(fig.data) == 4 * len(means)  # mean and fill traces added again

@needs_plotly
def test_plotly_violin(ntwk_set_zip: NetworkSet):
    fig = ntwk_set_zip.plot_violin("s_db", m=0, n=1, backend="plotly")
    freq = ntwk_set_zip[0].frequency
    (trace,) = fig.data
    assert trace.type == "violin"
    np.testing.assert_allclose(trace.y, np.array([ntwk.s_db[:, 0, 1] for ntwk in ntwk_set_zip]).ravel())
    np.testing.assert_allclose(trace.x, np.tile(freq.f_scaled, len(ntwk_set_zip)))
    np.testing.assert_allclose(trace.width, 0.75 * (freq.f_scaled[1] - freq.f_scaled[0]))
    assert (fig.layout.xaxis.title.text, fig.layout.yaxis.title.text) == (f"Frequency ({freq.unit})",
                                                                          rf.Network.Y_LABEL_DICT["db"])

@needs_plotly
def test_plotly_plot_it_all(ntwk1_dc: rf.Network):
    fig = ntwk1_dc.plot_it_all(backend="plotly")
    data = [t for t in fig.data if t.meta != "skrf_smith_chart"]
    assert len(data) == 4 * ntwk1_dc.nports**2
    assert {t.xaxis for t in data} == {"x", "x2", "x3", "x4"}
    assert (fig.layout.yaxis3.scaleanchor, fig.layout.yaxis4.scaleanchor) == ("x3", "x4")
    assert [t.name for t in fig.data if t.showlegend] == [t.name for t in data if t.xaxis == "x"]

@needs_plotly
@pytest.mark.parametrize("func", ["plot_rectangular", "plot_complex_rectangular", "plot_smith"])
def test_plotly_public_plot_functions(ntwk1: rf.Network, func: str, mpl_axes):
    args = (ntwk1.f, ntwk1.s_db[:, 0, 0]) if func == "plot_rectangular" else (ntwk1.s[:, 0, 0],)
    fig = getattr(rf.plotting, func)(*args, label="a", backend="plotly")
    getattr(rf.plotting, func)(*args, label="a", ax=mpl_axes)
    (trace,) = [t for t in fig.data if t.meta != "skrf_smith_chart"]
    np.testing.assert_allclose(trace.x, mpl_axes.lines[-1].get_xdata())
    np.testing.assert_allclose(trace.y, mpl_axes.lines[-1].get_ydata())
    assert trace.name == "a"
    n_traces = len(fig.data)
    assert getattr(rf.plotting, func)(*args, ax=fig) is fig
    assert len(fig.data) == n_traces + 1  # the smith chart is drawn once

@needs_plotly
def test_plotly_plot_polar():
    fig = rf.plotting.plot_complex_polar(np.array([1j, -2]), y_label="r", title="t", backend="plotly")
    (trace,) = fig.data
    assert (trace.type, trace.thetaunit) == ("scatterpolar", "radians")
    np.testing.assert_allclose(trace.r, [1, 2])
    np.testing.assert_allclose(trace.theta, [np.pi / 2, np.pi])
    assert (fig.layout.polar.radialaxis.title.text, fig.layout.title.text) == ("r", "t")

@needs_plotly
def test_plotly_subplot_params(ntwk1: rf.Network):
    fig = rf.plotting.subplot_params(ntwk1, backend="plotly")
    assert len(fig.data) == ntwk1.nports**2
    np.testing.assert_allclose(fig.data[1].y, ntwk1.s_db[:, 1, 0])  # in port_tuples order, as matplotlib
    assert {t.xaxis for t in fig.data} == {"x", "x2", "x3", "x4"}
    assert [a.text for a in fig.layout.annotations] == ["S11", "S21", "S12", "S22"]
    assert all(t.showlegend for t in fig.data)
    assert (fig.layout.width, fig.layout.height) == (800, 800)

def test_shade_bands(ntwk1: rf.Network, mpl_axes):
    ntwk1.plot_s_db(0, 0, ax=mpl_axes)
    rf.plotting.shade_bands(ntwk1.f[[0, 40, 80]], alpha=0.2)
    assert len(mpl_axes.collections) == 2

@needs_plotly
def test_plotly_plot_contour(monkeypatch):
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda fig: shown.append(fig))
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-0.6, 0.6, (2, 100))
    z = np.hypot(x - 0.1, y)
    gamma, value = rf.plotting.plot_contour(1e9, x, y, z, min0max1=0,
                                            title="t", backend="plotly")
    assert value == z.min()
    (fig,) = shown
    assert [t.type for t in fig.data] == ["contour"] + ["scatter"] * 4
    assert np.isnan(fig.data[0].z).any() and fig.layout.title.text == "t"

@pytest.fixture
def one_port_cal() -> rf.calibration.OnePort:
    """Overdetermined one-port calibration, with named standards for the error NetworkSets."""
    media = rf.media.DefinedGammaZ0(rf.Frequency(1, 10, 11, "GHz"))
    ideals = [media.short(name="short"), media.open(name="open"), media.load(0.2, name="load")]
    measured = [rf.Network(frequency=i.frequency, s=i.s + 1e-3 * (k + 1), name=i.name)
                for k, i in enumerate(ideals * 2)]
    cal = rf.calibration.OnePort(ideals=ideals * 2, measured=measured)
    cal.run()
    return cal

def test_calibration_plots_matplotlib(one_port_cal, mpl_axes):
    other = mpl_axes.figure.add_subplot(2, 1, 2)  # now the current axes, not the one passed in
    assert one_port_cal.plot_calibration_errors(ax=mpl_axes) is None
    assert len(mpl_axes.lines) == 3 and len(other.lines) == 0
    assert mpl_axes.get_title() == "S11" and mpl_axes.get_ylim() == (-100, 0)
    assert one_port_cal.plot_residuals(ax=other) is None
    assert len(other.lines) == len(one_port_cal.residual_ntwks)

@needs_plotly
def test_plotly_calibration_errors(one_port_cal, mpl_axes):
    one_port_cal.plot_calibration_errors(ax=mpl_axes)
    fig = one_port_cal.plot_calibration_errors(backend="plotly")
    assert_plotly_parity(fig, mpl_axes, x_scale=1e9)
    assert fig.layout.title.text == "S11" and tuple(fig.layout.yaxis.range) == (-100, 0)
    # global option, and adding to an existing figure
    rf.plotting.options.backend = "plotly"
    assert one_port_cal.plot_calibration_errors(ax=fig) is fig
    assert len(fig.data) == 6
    with pytest.raises(TypeError):
        one_port_cal.plot_calibration_errors(ax=mpl_axes, backend="plotly")

@needs_plotly
@pytest.mark.parametrize("attr", ["s_db", "s_smith"])
def test_plotly_calibration_residuals(one_port_cal, attr: str, mpl_axes):
    one_port_cal.plot_residuals(attr, ax=mpl_axes)
    fig = one_port_cal.plot_residuals(attr, backend="plotly")
    assert isinstance(fig, go.Figure)
    assert len([t for t in fig.data if t.name]) == len(one_port_cal.residual_ntwks)
    if attr == "s_db":
        assert_plotly_parity(fig, mpl_axes, x_scale=1e9)
