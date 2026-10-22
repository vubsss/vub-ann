"""Audit measured CSV artifacts and build the five-page Assignment 3 report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.utils import ImageReader


ROOT = Path(__file__).resolve().parents[2]
PAGE_W, PAGE_H = A4
MARGIN = 42
CONTENT_W = PAGE_W - 2 * MARGIN
INK = colors.HexColor("#183047")
MUTED = colors.HexColor("#526477")
TEAL = colors.HexColor("#087E8B")
LIGHT = colors.HexColor("#EDF4F6")
RULE = colors.HexColor("#CBD7DE")
FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
FONT_ITALIC = "Helvetica-Oblique"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def close(a: str | float, b: str | float, *, tolerance: float = 1e-8) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def load_and_audit(results_dir: Path) -> dict:
    raw = read_csv(results_dir / "results.csv")
    by_id = defaultdict(list)
    for row in raw:
        by_id[row["run_id"]].append(row)
    first = {key: next(row for row in rows if row["repeat_index"] == "1")
             for key, rows in by_id.items()}

    thresholds = read_csv(results_dir / "main_qps_thresholds.csv")
    memory = read_csv(results_dir / "main_build_memory.csv")
    plot_points = read_csv(results_dir / "main_plot_points.csv")
    difficulty = read_csv(results_dir / "query_difficulty.csv")
    synthetic = read_csv(results_dir / "cluster_imbalance_summary.csv")

    for row in thresholds:
        source = first[row["run_id"]]
        assert close(row["measured_recall"], source["recall_at_10"])
        if row["status"] == "reached":
            assert close(row["median_qps"], source["median_qps"])
    for row in memory:
        source = first[row["run_id"]]
        assert close(row["measured_recall"], source["recall_at_10"])
        if row["status"] == "reached":
            for field in ("build_seconds", "index_size_bytes", "bytes_per_vector"):
                assert close(row[field], source[field])
    for row in plot_points:
        source = first[row["run_id"]]
        assert close(row["recall_at_10"], source["recall_at_10"])
        assert close(row["median_qps"], source["median_qps"])
    assert len(plot_points) == 39

    assert len(difficulty) == 12
    for family in ("hnsw", "ivf_flat", "ivf_pq"):
        group = [r for r in difficulty if r["index_family"] == family]
        assert [int(r["bin"]) for r in group] == [1, 2, 3, 4]
        assert all(int(r["query_count"]) == 1250 for r in group)
        mean = sum(float(r["mean_recall_at_10"]) for r in group) / 4
        assert close(mean, first[group[0]["main_final_run_id"]]["recall_at_10"])
    assert len(synthetic) == 8
    assert len({r["query_sha256"] for r in synthetic}) == 1
    for row in synthetic:
        source = first[row["run_id"]]
        assert close(row["recall_at_10"], source["recall_at_10"])
        assert close(row["median_qps"], source["median_qps"])
        assert row["source_sha256"] == source["source_sha256"]
        assert (source["base_count"], source["dimensions"], source["reference_query_count"],
                source["seed"]) == ("96000", "32", "1024", "20261021")

    exact = {}
    for dataset in ("sift", "glove"):
        for mode in ("single", "batch"):
            matches = [r for r in raw if r["experiment"] == "exact_baseline"
                       and r["dataset"] == dataset and r["timing_mode"] == mode]
            assert len(matches) == 3 and all(int(r["reference_query_count"]) == 10000 for r in matches)
            exact[dataset, mode] = matches[0]
    for filename in ("sift_recall_qps.png", "glove_recall_qps.png",
                     "glove_query_difficulty.png", "cluster_imbalance.png"):
        assert (results_dir / "figures" / filename).is_file()
    # Narrative values below are rounded from these checked source rows. Fail
    # when a future rerun changes them, rather than silently publishing stale prose.
    for dataset, count, ties, single, batch in (
        ("sift", 1_000_000, 65, 36.1, 116.6),
        ("glove", 1_183_514, 1, 44.8, 116.6),
    ):
        one, many = exact[dataset, "single"], exact[dataset, "batch"]
        assert int(one["base_count"]) == count
        assert int(one["differing_id_set_queries"]) == ties
        assert close(one["exact_vs_stored_recall"], 1)
        assert close(one["stored_vs_exact_recall"], 1)
        assert fnum(one["median_qps"], 1) == fnum(single, 1)
        assert fnum(many["median_qps"], 1) == fnum(batch, 1)
    machine = exact["sift", "single"]
    assert "Core(TM) Ultra 7 155H" in machine["cpu_model"]
    assert (machine["logical_cpus"], machine["python_version"],
            machine["faiss_version"], machine["numpy_version"],
            machine["h5py_version"]) == ("22", "3.13.9", "1.15.1", "2.3.5", "3.15.1")
    assert round(int(machine["ram_bytes"]) / 2**30, 1) == 14.6
    expected_qps = {
        ("sift", "hnsw", "0.95"): 3219,
        ("sift", "ivf_flat", "0.95"): 943,
        ("glove", "hnsw", "0.95"): 652,
        ("glove", "ivf_flat", "0.95"): 142,
    }
    for (dataset, family, target), expected in expected_qps.items():
        match = next(r for r in thresholds if (r["dataset"], r["family"], r["target_recall"])
                     == (dataset, family, target))
        assert round(float(match["median_qps"])) == expected
    for dataset, build, search in (
        ("sift", {"M": 16, "efConstruction": 120}, {"efSearch": 64}),
        ("glove", {"M": 32, "efConstruction": 160}, {"efSearch": 512}),
    ):
        choice = next(r for r in thresholds if (r["dataset"], r["family"], r["target_recall"])
                      == (dataset, "hnsw", "0.95"))
        assert json.loads(choice["build_settings"]) == build
        assert json.loads(choice["search_settings"]) == search
    for dataset, family, build_s, bytes_vector in (
        ("sift", "hnsw", 208.6, 656.3),
        ("sift", "ivf_flat", 158.8, 522.1),
        ("glove", "hnsw", 509.9, 672.2),
        ("glove", "ivf_flat", 17.4, 408.4),
    ):
        entry = next(r for r in memory if (r["dataset"], r["family"]) == (dataset, family))
        assert fnum(entry["build_seconds"], 1) == fnum(build_s, 1)
        assert fnum(entry["bytes_per_vector"], 1) == fnum(bytes_vector, 1)
    for dataset, max_recall, pq_bytes, pq_build in (
        ("sift", 0.8735, 72.7, 65.4),
        ("glove", 0.8630, 58.4, 54.9),
    ):
        pq = [r for r in raw if r["experiment"] == "ann_final" and r["dataset"] == dataset
              and r["index_family"] == "ivf_pq" and r["repeat_index"] == "1"]
        assert fnum(max(float(r["recall_at_10"]) for r in pq), 4) == fnum(max_recall, 4)
        assert any(fnum(r["bytes_per_vector"], 1) == fnum(pq_bytes, 1)
                   and fnum(r["build_seconds"], 1) == fnum(pq_build, 1) for r in pq)
    by_bin = {(r["index_family"], int(r["bin"])): float(r["mean_recall_at_10"])
              for r in difficulty}
    for family, first_bin, last_bin in (
        ("hnsw", 0.9150, 0.9826),
        ("ivf_flat", 0.9636, 0.9865),
        ("ivf_pq", 0.8147, 0.9024),
    ):
        assert fnum(by_bin[family, 1], 4) == fnum(first_bin, 4)
        assert fnum(by_bin[family, 4], 4) == fnum(last_bin, 4)
    by_syn = {(r["index_family"], int(r["population_ratio"])): r for r in synthetic}
    for family, control, extreme in (("hnsw", 0.9068, 0.9039),
                                     ("ivf_flat", 0.9721, 0.9418)):
        assert fnum(by_syn[family, 1]["recall_at_10"], 4) == fnum(control, 4)
        assert fnum(by_syn[family, 64]["recall_at_10"], 4) == fnum(extreme, 4)
    ivf0, ivf64 = by_syn["ivf_flat", 1], by_syn["ivf_flat", 64]
    assert fnum(ivf0["common_quartile_recall_at_10"], 4) == "0.9262"
    assert fnum(ivf64["common_quartile_recall_at_10"], 4) == "0.7672"
    assert fnum(ivf0["rare_quartile_recall_at_10"], 4) == "1.0000"
    assert fnum(ivf64["rare_quartile_recall_at_10"], 4) == "1.0000"
    c0, c64 = json.loads(ivf0["ivf_centroid_counts_json"]), json.loads(ivf64["ivf_centroid_counts_json"])
    assert (sum(c0[-4:]), sum(c64[-4:])) == (22, 45)
    assert (round(float(ivf0["common_probed_candidates"])),
            round(float(ivf64["common_probed_candidates"]))) == (5625, 6644)
    return {
        "raw": raw, "first": first, "thresholds": thresholds, "memory": memory,
        "difficulty": difficulty, "synthetic": synthetic, "exact": exact,
        "figures": results_dir / "figures",
    }


def setup_fonts() -> None:
    global FONT_REGULAR, FONT_BOLD, FONT_ITALIC
    base = Path("/usr/share/fonts/truetype/lato")
    if all((base / name).is_file() for name in
           ("Lato-Regular.ttf", "Lato-Bold.ttf", "Lato-Italic.ttf")):
        FONT_REGULAR, FONT_BOLD, FONT_ITALIC = "Lato", "Lato-Bold", "Lato-Italic"
        pdfmetrics.registerFont(TTFont(FONT_REGULAR, str(base / "Lato-Regular.ttf")))
        pdfmetrics.registerFont(TTFont(FONT_BOLD, str(base / "Lato-Bold.ttf")))
        pdfmetrics.registerFont(TTFont(FONT_ITALIC, str(base / "Lato-Italic.ttf")))
    pdfmetrics.registerFontFamily(FONT_REGULAR, normal=FONT_REGULAR,
                                  bold=FONT_BOLD, italic=FONT_ITALIC)


def paragraph(c: canvas.Canvas, html: str, y: float, *, size: float = 9.15,
              leading: float = 12.35, gap: float = 8, width: float = CONTENT_W,
              color: colors.Color = INK) -> float:
    style = ParagraphStyle("body", fontName=FONT_REGULAR, fontSize=size, leading=leading,
                           textColor=color, alignment=TA_LEFT)
    item = Paragraph(html, style)
    _, height = item.wrap(width, PAGE_H)
    item.drawOn(c, MARGIN, y - height)
    return y - height - gap


def label(c: canvas.Canvas, text: str, y: float) -> float:
    c.setFillColor(TEAL)
    c.setFont(FONT_BOLD, 9.4)
    c.drawString(MARGIN, y, text.upper())
    return y - 13


def question(c: canvas.Canvas, text: str, y: float) -> float:
    c.setFillColor(TEAL)
    c.setFont(FONT_BOLD, 10.1)
    c.drawString(MARGIN, y, text.upper())
    return y - 20


def discussion(c: canvas.Canvas, html: str, y: float, *, gap: float = 15) -> float:
    return paragraph(c, html, y, size=10.1, leading=14.2, gap=gap)


def header(c: canvas.Canvas, section: str, title: str, page: int,
           subtitle: str | None = None) -> float:
    c.setFillColor(TEAL)
    c.setFont(FONT_BOLD, 8)
    c.drawString(MARGIN, PAGE_H - 36, section.upper())
    c.setFillColor(INK)
    c.setFont(FONT_BOLD, 21 if page == 1 else 18)
    c.drawString(MARGIN, PAGE_H - 65, title)
    y = PAGE_H - 80
    if subtitle:
        y = paragraph(c, subtitle, y, size=9.1, leading=11.7, gap=5, color=MUTED)
    c.setStrokeColor(RULE)
    c.setLineWidth(0.7)
    c.line(MARGIN, y, PAGE_W - MARGIN, y)
    return y - 14


def footer(c: canvas.Canvas, page: int) -> None:
    c.setStrokeColor(RULE)
    c.line(MARGIN, 35, PAGE_W - MARGIN, 35)
    c.setFillColor(MUTED)
    c.setFont(FONT_REGULAR, 7.5)
    c.drawString(MARGIN, 23, "CS4.406  |  Assignment 3  |  Measurements and code in accompanying archive")
    c.drawRightString(PAGE_W - MARGIN, 23, f"{page} / 5")


def table(c: canvas.Canvas, data: list[list[str]], widths: list[float], y: float,
          *, row_height: float = 18.5, font_size: float = 8.15) -> float:
    rows = len(data)
    item = Table(data, colWidths=widths, rowHeights=[row_height] * rows)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTNAME", (0, 1), (-1, -1), FONT_REGULAR),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, TEAL),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
    ]
    for row in range(2, rows, 2):
        style.append(("BACKGROUND", (0, row), (-1, row), LIGHT))
    item.setStyle(TableStyle(style))
    _, height = item.wrap(CONTENT_W, PAGE_H)
    item.drawOn(c, MARGIN, y - height)
    return y - height - 9


def chart(c: canvas.Canvas, path: Path, y: float, *, width: float,
          caption: str, bottom_gap: float = 11) -> float:
    image = ImageReader(str(path))
    px_w, px_h = image.getSize()
    height = width * px_h / px_w
    x = (PAGE_W - width) / 2
    c.drawImage(image, x, y - height, width=width, height=height)
    y = y - height - 5
    y = paragraph(c, caption, y, size=7.9, leading=10.2, gap=bottom_gap, color=MUTED)
    return y


def fnum(value: str | float, digits: int = 0) -> str:
    return f"{float(value):,.{digits}f}"


def threshold(data: dict, dataset: str, family: str, target: float) -> dict[str, str]:
    return next(r for r in data["thresholds"] if r["dataset"] == dataset
                and r["family"] == family and close(r["target_recall"], target))


def build_report(output: Path, results_dir: Path) -> None:
    d = load_and_audit(results_dir)
    setup_fonts()
    output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output), pagesize=A4, pageCompression=1)
    c.setTitle("Benchmarking approximate nearest-neighbour search")
    c.setAuthor("CS4.406 Assignment 3")
    c.setSubject("Measured ANN trade-offs on SIFT-1M, GloVe-100, and seeded synthetic clusters")

    # Page 1 - protocol and exact baseline.
    y = header(c, "method and baseline", "Benchmarking ANN search", 1,
               "SIFT-1M and GloVe-100  |  HNSW, IVF-Flat, IVF-PQ  |  recall@10 against single-query throughput")
    y = label(c, "Method", y)
    y = paragraph(c, "SIFT has <b>1,000,000</b> 128-D Euclidean vectors; GloVe has <b>1,183,514</b> 100-D vectors. We unit-normalize GloVe base and queries, then use squared Euclidean distance throughout. Original queries 1-5,000 are for tuning; all reported public-data results use held-out queries 5,001-10,000. Full-size sweeps chose build and search settings from 1,000 tuning queries; final searches use all base vectors and three timing repetitions.", y)
    y = paragraph(c, "FAISS <b>IndexFlatL2</b> supplies exact top ten. Our recall function counts distinct retrieved IDs whose true source-vector distance is no greater than the exact tenth-neighbour distance, including ties. Against all 10,000 stored references, both datasets give tie-aware recall 1.0 in both directions; only <b>65 SIFT</b> and <b>1 GloVe</b> top-ten ID sets differ, all at tie boundaries.", y)
    y = paragraph(c, "Machine: Intel Core Ultra 7 155H, 22 logical CPUs, 14.6 GiB RAM. Python 3.13.9, FAISS 1.15.1, NumPy 2.3.5, h5py 3.15.1. Timed search uses one FAISS/OpenMP thread and one query per call; full HNSW construction uses four threads. Build time includes training and insertion; index size is serialized FAISS bytes.", y, gap=10)
    y = label(c, "Exact search: median over 3 repetitions", y)
    ex = d["exact"]
    exact_rows = [["Dataset", "Vectors", "Single QPS", "Batch QPS", "Single ms/q"]]
    for dataset, name, n in (("sift", "SIFT-1M", 1_000_000), ("glove", "GloVe-100", 1_183_514)):
        single, batch = ex[dataset, "single"], ex[dataset, "batch"]
        exact_rows.append([name, fnum(n), fnum(single["median_qps"], 1),
                           fnum(batch["median_qps"], 1),
                           fnum(1000 / float(single["median_qps"]), 1)])
    y = table(c, exact_rows, [94, 100, 100, 100, CONTENT_W - 394], y, row_height=19.5)
    y = paragraph(c, "Batch exact search reuses vector blocks and pays the Python-to-FAISS call overhead once. It is therefore faster than one-at-a-time exact search even with one thread; the ANN comparisons below use only the single-query baseline.", y, size=8.7, leading=11.3, gap=16)
    y = label(c, "SIFT-1M trade-off", y)
    y = chart(c, d["figures"] / "sift_recall_qps.png", y, width=455,
              caption="Figure 1. Held-out recall@10 versus single-query QPS; curves show measured Pareto points, with exact Flat at recall 1.0.")
    assert y > 45, f"page 1 overflow: {y}"
    footer(c, 1)
    c.showPage()

    # Page 2 - GloVe curve, threshold table, and build/memory table.
    y = header(c, "main benchmark", "Recall, speed and index cost", 2)
    y = chart(c, d["figures"] / "glove_recall_qps.png", y, width=445,
              caption="Figure 2. GloVe-100 held-out recall@10 versus single-query QPS; exact Flat is the reference point.", bottom_gap=16)
    y = label(c, "QPS at frozen tuning selections", y)
    threshold_rows = [["Index", "SIFT @0.90", "SIFT @0.95", "GloVe @0.90", "GloVe @0.95"]]
    for family, name in (("hnsw", "HNSW"), ("ivf_flat", "IVF-Flat"),
                         ("ivf_pq", "IVF-PQ"), ("IndexFlatL2", "Exact Flat")):
        cells = [name]
        for dataset in ("sift", "glove"):
            for target in (0.90, 0.95):
                row = threshold(d, dataset, family, target)
                cells.append(fnum(row["median_qps"]) if row["status"] == "reached" else "not reached")
        threshold_rows.append(cells)
    y = table(c, threshold_rows, [101, 102, 102, 102, CONTENT_W - 407], y, row_height=18.5)
    y = paragraph(c, "A threshold entry reports the setting frozen on tuning data, not an interpolation. IVF-PQ reached at most <b>0.8735</b> on SIFT and <b>0.8630</b> on GloVe in the tested eight-bit grids; it missed even 0.90.", y, size=8.4, leading=10.8, gap=16)
    y = label(c, "At recall 0.95: build time and serialized size", y)
    mem_rows = [["Dataset", "Index", "Recall", "Build s", "Size MiB", "Bytes/vector"]]
    for dataset, name in (("sift", "SIFT"), ("glove", "GloVe")):
        for family, label_name in (("hnsw", "HNSW"), ("ivf_flat", "IVF-Flat"), ("ivf_pq", "IVF-PQ")):
            row = next(r for r in d["memory"] if r["dataset"] == dataset and r["family"] == family)
            if row["status"] == "reached":
                mem_rows.append([name, label_name, fnum(row["measured_recall"], 4),
                                 fnum(row["build_seconds"], 1),
                                 fnum(int(row["index_size_bytes"]) / 2**20, 1),
                                 fnum(row["bytes_per_vector"], 1)])
            else:
                mem_rows.append([name, label_name, "<0.95", "--", "--", "--"])
    y = table(c, mem_rows, [66, 82, 67, 68, 82, CONTENT_W - 365], y, row_height=18.2)
    y = paragraph(c, "HNSW is fastest at 0.95 on both datasets: <b>3,219 versus 943 QPS</b> for SIFT and <b>652 versus 142 QPS</b> for GloVe (HNSW versus IVF-Flat). The tested IVF-PQ remained below target even with every coarse list probed; compressed distance ranking can still misorder candidates.", y, size=8.55, leading=11.3, gap=0)
    assert y > 45, f"page 2 overflow: {y}"
    footer(c, 2)
    c.showPage()

    # Page 3 - Part 3 query difficulty.
    y = header(c, "part 3 / standard data", "Which GloVe queries are hard?", 3)
    y = paragraph(c, "<b>Preregistered hypothesis (commit 82d9cdc):</b> local contrast should make high-contrast queries easier for all three indexes, with the largest bin-4 minus bin-1 gain for IVF-PQ. For each held-out query, score = (d10 - d1) / d10 using true squared L2 distances to exact neighbours. Sort by score (original query row breaks ties) into four bins of 1,250; bin 1 has the smallest contrast.", y, gap=10)
    y = chart(c, d["figures"] / "glove_query_difficulty.png", y, width=445,
              caption="Figure 3. Recall@10 by exact-neighbour contrast, using frozen full-size GloVe settings.", bottom_gap=16)
    y = label(c, "Held-out recall by bin", y)
    q_rows = [["Bin", "Contrast range", "Count", "HNSW", "IVF-Flat", "IVF-PQ"]]
    for bin_no in range(1, 5):
        group = {r["index_family"]: r for r in d["difficulty"] if int(r["bin"]) == bin_no}
        h = group["hnsw"]
        q_rows.append([str(bin_no), f"{float(h['score_min']):.4f}-{float(h['score_max']):.4f}",
                       "1,250", *(fnum(group[family]["mean_recall_at_10"], 4)
                                  for family in ("hnsw", "ivf_flat", "ivf_pq"))])
    y = table(c, q_rows, [44, 126, 65, 90, 94, CONTENT_W - 419], y, row_height=18.5)
    y = paragraph(c, "<b>Outcome:</b> all three improve from bin 1 to 4; endpoint gains are HNSW +0.0676, IVF-Flat +0.0229 and IVF-PQ +0.0877. IVF-Flat dips slightly from bin 3 to 4, so the monotonic prediction holds only for HNSW and IVF-PQ. Close neighbours may be displaced by approximate ranking, particularly by PQ distance error; probing all GloVe IVF-PQ lists still leaves recall at 0.8630.", y, gap=7)
    y = paragraph(c, "The score measures spread within the top ten, not the tenth-to-eleventh gap, centroid-boundary position or graph reachability. It describes association, not a causal effect. HNSW and IVF-Flat preserve source vectors; PQ compresses them. The three settings have different search effort, so gap sizes are not controlled comparisons of intrinsic sensitivity.", y, gap=0)
    assert y > 45, f"page 3 overflow: {y}"
    footer(c, 3)
    c.showPage()

    # Page 4 - Part 4 is contained entirely on one page.
    y = header(c, "part 4 / generated data", "When clusters become unbalanced", 4)
    y = paragraph(c, "<b>Preregistered hypothesis (commit 379f0d9):</b> imbalance should reduce IVF-Flat recall more than HNSW, especially for rare-cluster queries. Seed 20261021 produces 96,000 32-D vectors around 16 fixed Gaussian centres and 1,024 separate, fixed queries (64 per cluster). Only the base population ratio changes: 1:1 control, then 4:1, 16:1 and 64:1. Base noise, insertion order, centre spread and index settings stay fixed.", y, gap=7)
    y = chart(c, d["figures"] / "cluster_imbalance.png", y, width=498,
              caption="Figure 4. Exact-reference recall@10 and median one-query-at-a-time QPS; three repetitions of 20 passes at each ratio.", bottom_gap=16)
    y = label(c, "Fixed search settings and observed recall", y)
    syn_rows = [["Ratio", "Min-max base/cluster", "HNSW recall", "IVF-Flat recall", "IVF-Flat QPS"]]
    for ratio in (1, 4, 16, 64):
        h = next(r for r in d["synthetic"] if r["index_family"] == "hnsw" and int(r["population_ratio"]) == ratio)
        i = next(r for r in d["synthetic"] if r["index_family"] == "ivf_flat" and int(r["population_ratio"]) == ratio)
        syn_rows.append([f"{ratio}:1", f"{h['cluster_min_count']}-{h['cluster_max_count']}",
                         fnum(h["recall_at_10"], 4), fnum(i["recall_at_10"], 4), fnum(i["median_qps"])])
    y = table(c, syn_rows, [63, 132, 97, 113, CONTENT_W - 405], y, row_height=18.2)
    y = paragraph(c, "HNSW (M=16, efConstruction=100, efSearch=32) changes from 0.9068 to 0.9039; IVF-Flat (nlist=64, nprobe=4) falls from 0.9721 to 0.9418. The predicted <b>larger IVF-Flat decline</b> occurred, but the rare-cluster explanation failed: its four smallest clusters kept recall 1.0000, while the four largest fell from 0.9262 to 0.7672.", y, gap=6)
    y = paragraph(c, "Assigning learned centroids to their nearest generating centre, the four largest clusters drew 22 then 45 of 64; fixed nprobe=4 covers less of a populous cluster even as mean scanned candidates rose from 5,625 to 6,644. This supports a partition-coverage explanation, without locating every missed neighbour. HNSW graph paths were not inspected. The seed reproduced all data hashes and recall on a second run; timing varies. Dimension, spread, query mix and search effort were not varied.", y, size=8.75, leading=11.6, gap=0)
    assert y > 45, f"page 4 overflow: {y}"
    footer(c, 4)
    c.showPage()

    # Page 5 - Part 5 questions and limitations.
    y = header(c, "discussion / decisions", "What the measurements imply", 5)
    y = question(c, "1. Fastest at recall 0.95", y)
    y = discussion(c, "HNSW wins on both datasets, with the measured pairs above: SIFT 3,219 versus 943 QPS and GloVe 652 versus 142 QPS against IVF-Flat. Its graph search follows promising links instead of scanning the full base, while keeping exact vector distances. This is a comparison of frozen tuned settings on one machine, not a universal ranking.", y)
    y = question(c, "2. Build and memory cost; what PQ buys", y)
    y = discussion(c, "At 0.95, HNSW needs 208.6 s and 656.3 bytes/vector on SIFT, and 509.9 s and 672.2 bytes/vector on GloVe. IVF-Flat needs 158.8 s / 522.1 bytes/vector and 17.4 s / 408.4 bytes/vector, respectively. The best tested IVF-PQ builds use only <b>72.7</b> and <b>58.4</b> bytes/vector (build 65.4 and 54.9 s), but top out at recall 0.8735 and 0.8630. PQ trades memory for quantized ranking error; scanning all coarse lists did not restore 0.90.", y)
    y = question(c, "3. When headline numbers mislead", y)
    y = discussion(c, "GloVe bin-1 recall is below bin-4 recall for every index (for IVF-PQ, 0.8147 versus 0.9024), so an overall mean hides hard queries. In synthetic data, overall IVF-Flat recall fell only 0.0303 across imbalance, yet large-cluster recall fell 0.1590. Query mix can therefore conceal who loses accuracy; the rare-cluster prediction was wrong.", y)
    y = question(c, "4. 150,000 or 10 million vectors; 100 ms and 16 GB", y)
    sift_exact_150 = 1000 / float(ex["sift", "single"]["median_qps"]) * 0.15
    glove_exact_150 = 1000 / float(ex["glove", "single"]["median_qps"]) * (150_000 / 1_183_514)
    sift_exact_10m = 1000 / float(ex["sift", "single"]["median_qps"]) * 10
    glove_exact_10m = 1000 / float(ex["glove", "single"]["median_qps"]) * (10_000_000 / 1_183_514)
    y = discussion(c, f"<b>150K:</b> choose exact Flat when perfect recall matters. A linear extrapolation of measured one-at-a-time exact time gives about {sift_exact_150:.1f} ms/query (SIFT) or {glove_exact_150:.1f} ms (GloVe), well below 100 ms. <b>10M:</b> exact extrapolates to about {sift_exact_10m:.0f} and {glove_exact_10m:.0f} ms, beyond budget; start with frozen HNSW 0.95 settings (SIFT M=16, efConstruction=120, efSearch=64; GloVe M=32, efConstruction=160, efSearch=512). Serialized size projected linearly is roughly 6.6-6.7 GB, within 16 GB. Re-tune and measure at 10M: recall, latency and peak build RAM were not tested there.", y)
    y = question(c, "5. Two benchmark limitations and whether tested", y)
    y = discussion(c, "<b>Query distribution:</b> a single overall recall can hide difficult queries or cluster-specific misses. <i>Tested</i> with GloVe contrast bins and fixed-mix synthetic clusters. <b>Scale and resource extrapolation:</b> index overhead, training quality, caches and build peak RAM may change from about 1M to 10M; linear projections can mislead. <i>Not tested</i> at 10M. Single-thread search also excludes concurrent serving, which was not tested.", y, gap=12)
    y = paragraph(c, "<b>Traceability.</b> Main figures and tables map to run IDs in results/results.csv; Part 3 bin rows link to their final runs; Part 4 uses eight seeded settings with three timing rows each. Source and exact-neighbour hashes are recorded. Hypothesis commits predate experiment-result commits. All reported public-data settings were frozen from tuning queries.", y, size=8.25, leading=10.8, gap=0, color=MUTED)
    assert y > 45, f"page 5 overflow: {y}"
    footer(c, 5)
    c.showPage()
    c.save()
    print(f"Wrote {output} (5 pages; CSV provenance checks passed)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "report.pdf")
    args = parser.parse_args()
    build_report(args.output, args.results_dir)


if __name__ == "__main__":
    main()
