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


# Plotting backend selection (skrf.options.plotting.backend) and the plotly backend

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

needs_plotly = pytest.mark.skipif(go is None, reason="plotly not installed")


@pytest.fixture(autouse=True)
def reset_plotting_backend():
    yield
    rf.options.plotting.backend = "matplotlib"

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
    assert rf.options.plotting.backend == "matplotlib"
    rf.options.plotting.backend = "plotly"
    assert rf.options.plotting.backend == "plotly"
    with pytest.raises(ValueError, match="matplotlib.*plotly"):
        rf.options.plotting.backend = "bokeh"
    assert rf.options.plotting.backend == "plotly"

def test_plotly_missing_raises_import_error(ntwk1: rf.Network, monkeypatch):
    for mod in [m for m in sys.modules if m.split(".")[0] == "plotly"]:
        monkeypatch.delitem(sys.modules, mod)
    monkeypatch.setitem(sys.modules, "plotly", None)
    with pytest.raises(ImportError, match=r"scikit-rf\[plotly\]"):
        ntwk1.plot_s_db(backend="plotly")

@pytest.mark.parametrize("call", [
    lambda n, ns: n.plot_s_smith(),
    lambda n, ns: n.plot_s_polar(),
    lambda n, ns: n.plot_it_all(),
    lambda n, ns: ns.plot_uncertainty_bounds_s_db(),
    lambda n, ns: ns.plot_violin("s_db"),
], ids=["plot_s_smith", "plot_s_polar", "plot_it_all", "plot_uncertainty_bounds_s_db", "plot_violin"])
def test_plotly_not_implemented_does_not_fall_back(ntwk1: rf.Network, ntwk_set_zip: NetworkSet, call):
    rf.options.plotting.backend = "plotly"
    with pytest.raises(NotImplementedError, match="plotly"):
        call(ntwk1, ntwk_set_zip)

def test_matplotlib_unchanged_under_global_plotly(ntwk1: rf.Network, ntwk_set_zip: NetworkSet, mpl_axes):
    rf.options.plotting.backend = "plotly"
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
        "import skrf as rf; rf.options.plotting.backend = 'plotly'; n = rf.data.ring_slot; "
        "n.plot_s_db(); n.plot_s_time_db(); n.frequency.plot(n.s_db); n.plot_passivity()")
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
], ids=["plot_passivity", "plot_reciprocity2", "plot_s_error", "Frequency.plot", "NetworkSet.plot_s_db",
        "plot_logsigma"])
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
