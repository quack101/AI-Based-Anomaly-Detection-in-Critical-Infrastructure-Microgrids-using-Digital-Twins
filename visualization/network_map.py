"""
Feeder map with a meter set overlaid: static PNG (matplotlib) and an
interactive HTML (plotly, hover for bus / load / kW / phase / depth).

  python -m visualization.network_map --set H45
  python -m visualization.network_map --set A --no-html

Encoding
  filled circle, colour = phase (A/B/C), thick ring = three-phase meter,
      area ~ meter kW            metered load (real P,Q,|V| channels)
  hollow grey circle             unmetered load (pseudo-measurement)
  small dot                      bus with no load (zero-injection rows)
  star = source, square = regulator bus, triangle = capacitor bus,
  diamond = switch bus; dark line = main feeder, dashed = regulator
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from visualization.feeder_data import (
    FIGURES_DIR, load_feeder, load_meter_set, phase_kw,
)

PHASE_COLOUR = {"A": "#c0392b", "B": "#1f77b4", "C": "#2e9e5b", "ABC": "#444444"}


def load_phase_key(l):
    return "ABC" if (l.get("phases", 1) == 3 or len(str(l["phase"])) > 1) else str(l["phase"])


def index_loads(feeder, meters):
    """-> (meter by bus, loads by bus). One meter per bus by construction."""
    m_by_bus = {str(m["bus"]): m for m in meters}
    by_bus = {}
    for l in feeder.loads:
        by_bus.setdefault(str(l["bus"]), []).append(l)
    return m_by_bus, by_bus


def metered_load(m, by_bus):
    b = str(m["bus"])
    return next((l for l in by_bus.get(b, []) if l["name"] == m["load_name"]), by_bus[b][0])


def draw_network(ax, feeder, meters, with_labels=True):
    """Draw the feeder and one meter set onto `ax`. Returns legend handles."""
    m_by_bus, by_bus = index_loads(feeder, meters)
    mb = set(m_by_bus)
    ax.set_aspect("equal")
    ax.axis("off")

    for a, b in feeder.edges:
        (x1, y1), (x2, y2) = feeder.pos[a], feeder.pos[b]
        if frozenset((a, b)) in feeder.regulator_edges:
            ax.plot([x1, x2], [y1, y2], "--", color="#8e44ad", lw=1.6, zorder=1)
        else:
            ax.plot([x1, x2], [y1, y2], "-", color="#bbb", lw=1.0, zorder=1)
    # main feeder: the ordered path placement uses (largest downstream load at each step);
    # a hop that skips a regulator/switch node is drawn straight across it
    ax.plot([feeder.pos[b][0] for b in feeder.backbone], [feeder.pos[b][1] for b in feeder.backbone],
            "-", color="#555", lw=2.4, zorder=1.5, solid_capstyle="round")

    load_buses = set(by_bus)
    for b in feeder.buses:
        if b not in load_buses and b != feeder.source:
            x, y = feeder.pos[b]
            ax.scatter(x, y, s=10, color="#999", zorder=2)
    for b in load_buses - mb:
        x, y = feeder.pos[b]
        ax.scatter(x, y, s=46, facecolors="white", edgecolors="#777", linewidths=1.1, zorder=3)
    for b, m in m_by_bus.items():
        key = load_phase_key(metered_load(m, by_bus))
        x, y = feeder.pos[b]
        ax.scatter(x, y, s=40 + 9 * m["base_kw"] ** 0.9, color=PHASE_COLOUR[key],
                   edgecolors="black" if key == "ABC" else "white",
                   linewidths=2.2 if key == "ABC" else 0.8, zorder=4)
    for b in feeder.buses:
        x, y = feeder.pos[b]
        if b == feeder.source:
            ax.scatter(x, y, marker="*", s=420, color="#f1c40f", edgecolors="black", zorder=6)
        elif b in feeder.regulator_buses and b not in feeder.switch_buses:
            ax.scatter(x, y, marker="s", s=70, color="#8e44ad", edgecolors="white", zorder=5)
        elif b in feeder.capacitor_buses:
            ax.scatter(x, y, marker="^", s=80, color="#e67e22", edgecolors="white", zorder=5)
        elif b in feeder.switch_buses and b not in load_buses:
            ax.scatter(x, y, marker="D", s=42, color="#16a085", edgecolors="white", zorder=5)
    if with_labels:
        for m in meters:
            x, y = feeder.pos[str(m["bus"])]
            ax.annotate(m["meter_id"][1:], (x, y), xytext=(4, 5), textcoords="offset points",
                        fontsize=6, color="#222", zorder=7)

    return [
        Line2D([], [], marker="o", ls="", color=PHASE_COLOUR["A"], label="metered, phase A"),
        Line2D([], [], marker="o", ls="", color=PHASE_COLOUR["B"], label="metered, phase B"),
        Line2D([], [], marker="o", ls="", color=PHASE_COLOUR["C"], label="metered, phase C"),
        Line2D([], [], marker="o", ls="", color=PHASE_COLOUR["ABC"], markeredgecolor="black",
               label="metered, three-phase"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#777",
               label="unmetered load (pseudo-measurement)"),
        Line2D([], [], marker=".", ls="", color="#999", label="no load (zero-injection)"),
        Line2D([], [], marker="*", ls="", color="#f1c40f", markeredgecolor="black", markersize=14,
               label="source / substation head"),
        Line2D([], [], marker="s", ls="", color="#8e44ad", label="regulator bus"),
        Line2D([], [], marker="^", ls="", color="#e67e22", label="capacitor bus"),
        Line2D([], [], marker="D", ls="", color="#16a085", label="switch bus (no load)"),
        Line2D([], [], color="#555", lw=2.2, label="main feeder"),
        Line2D([], [], color="#8e44ad", lw=1.6, ls="--", label="regulator"),
    ]


def draw_static(feeder, meters, set_name, out_png):
    fig = plt.figure(figsize=(17, 11.5))
    ax = fig.add_axes([0.01, 0.11, 0.70, 0.84])
    handles = draw_network(ax, feeder, meters)
    _, by_bus = index_loads(feeder, meters)
    n_m = len(meters)
    ax.set_title(f"IEEE-123 feeder - meter set {set_name}: {n_m} metered of {len(by_bus)} load buses "
                 f"({len(feeder.loads)} loads)", fontsize=14, loc="left")
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.0), fontsize=9,
               frameon=False, ncol=4)

    names = {m["load_name"] for m in meters}
    allk, mk = phase_kw(feeder.loads), phase_kw(feeder.loads, names)
    ta, tm = sum(allk.values()), sum(mk.values())
    ax2 = fig.add_axes([0.745, 0.60, 0.235, 0.30])
    xs = range(3)
    ax2.bar([i - 0.2 for i in xs], [100 * allk[p] / ta for p in "ABC"], 0.4, color="#bbb",
            label="all 91 loads")
    ax2.bar([i + 0.2 for i in xs], [100 * mk[p] / tm for p in "ABC"], 0.4,
            color=[PHASE_COLOUR[p] for p in "ABC"], label=f"{n_m} metered")
    ax2.set_xticks(list(xs), ["A", "B", "C"])
    ax2.set_ylabel("share of kW (%)")
    ax2.set_title("Phase balance of the metered kW", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    ax3 = fig.add_axes([0.745, 0.20, 0.235, 0.28])
    depths = [m["depth"] for m in meters]
    ax3.hist(depths, bins=range(0, max(depths) + 3), color="#34495e", rwidth=0.85)
    ax3.set_xlabel("hops from source")
    ax3.set_ylabel("meters")
    ax3.set_title("Meter depth in the feeder", fontsize=10)
    ax3.spines[["top", "right"]].set_visible(False)
    fig.text(0.745, 0.06,
             f"on main feeder: {sum(1 for m in meters if m.get('on_backbone'))} of {n_m}\n"
             f"branches reached: {len({m['branch'] for m in meters})}\n"
             f"metered kW: {tm:,.0f} of {ta:,.0f} ({100 * tm / ta:.0f}%)", fontsize=10, va="bottom")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def draw_html(feeder, meters, set_name, out_html):
    import plotly.graph_objects as go
    m_by_bus, by_bus = index_loads(feeder, meters)
    mb = set(m_by_bus)
    fig = go.Figure()
    xs, ys = [], []
    for a, b in feeder.edges:
        xs += [feeder.pos[a][0], feeder.pos[b][0], None]
        ys += [feeder.pos[a][1], feeder.pos[b][1], None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#bbb", width=1),
                             hoverinfo="skip", name="lines"))

    def pts(buses, name, marker, text):
        buses = list(buses)
        fig.add_trace(go.Scatter(
            x=[feeder.pos[b][0] for b in buses], y=[feeder.pos[b][1] for b in buses],
            mode="markers", name=name, marker=marker,
            text=[text(b) for b in buses], hoverinfo="text"))

    load_buses = set(by_bus)
    pts([b for b in feeder.buses if b not in load_buses and b != feeder.source], "no-load bus",
        dict(size=5, color="#999"), lambda b: f"bus {b}<br>no load (zero-injection)")
    pts(load_buses - mb, "unmetered load",
        dict(size=9, color="white", line=dict(color="#777", width=1.5)),
        lambda b: f"bus {b}<br>" + "<br>".join(
            f"{l['name']}: {l['base_kw']:.0f} kW, phase {l['phase']}" for l in by_bus[b])
        + "<br>pseudo-measurement")
    for key in ("A", "B", "C", "ABC"):
        bs = [b for b, m in m_by_bus.items() if load_phase_key(metered_load(m, by_bus)) == key]
        pts(bs, f"metered {key}",
            dict(size=[8 + 0.55 * m_by_bus[b]["base_kw"] ** 0.9 for b in bs],
                 color=PHASE_COLOUR[key], line=dict(color="white", width=1)),
            lambda b: (f"<b>{m_by_bus[b]['meter_id']}</b> {m_by_bus[b]['load_name']} @ bus {b}<br>"
                       f"{m_by_bus[b]['base_kw']:.0f} kW, {m_by_bus[b]['size_class']}<br>"
                       f"depth {m_by_bus[b]['depth']}, branch {m_by_bus[b]['branch']}, "
                       f"{'main feeder' if m_by_bus[b].get('on_backbone') else 'lateral'}"))
    pts(sorted(feeder.regulator_buses & set(feeder.buses)), "regulator bus",
        dict(size=10, symbol="square", color="#8e44ad"), lambda b: f"bus {b}<br>regulator")
    pts(sorted(feeder.capacitor_buses & set(feeder.buses)), "capacitor bus",
        dict(size=11, symbol="triangle-up", color="#e67e22"), lambda b: f"bus {b}<br>capacitor")
    pts([feeder.source], "source",
        dict(size=18, symbol="star", color="#f1c40f", line=dict(color="black", width=1)),
        lambda b: f"bus {b}<br>substation head (HEAD_P/Q/V channels)")
    fig.update_layout(title=f"IEEE-123 feeder - meter set {set_name} ({len(meters)} meters)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                      plot_bgcolor="white", height=820, legend=dict(itemsizing="constant"))
    fig.write_html(out_html, include_plotlyjs="cdn")


def main(argv=None):
    from sensors.meter_set import METER_SETS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", default="H45", choices=sorted(METER_SETS))
    p.add_argument("--no-html", action="store_true")
    a = p.parse_args(argv)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    feeder = load_feeder()
    meters = load_meter_set(a.set)
    png = FIGURES_DIR / f"network_{a.set}.png"
    draw_static(feeder, meters, a.set, png)
    print("wrote", png)
    if not a.no_html:
        html = FIGURES_DIR / f"network_{a.set}.html"
        draw_html(feeder, meters, a.set, html)
        print("wrote", html)
    if feeder.missing_coords:
        print("note: no coordinates for", feeder.missing_coords, "(drawn beside a neighbour)")


if __name__ == "__main__":
    main()
