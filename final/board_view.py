# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "py-draughts>=1.6.4",
# ]
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Board view — FEN + move visualizer

    Type a FEN, type a move (e.g. `34-29` for a slide, `21x17` for a capture,
    `21x14x7` for a multi-jump), and see the before/after boards side by
    side. The "after" board highlights the move squares; the green arrow on
    top is positioned by the four xy sliders below — they default to the
    pixel centers of the move's tail/head squares when you change the move.
    """)
    return


@app.cell
def _():
    import draughts
    import draughts.svg as draughts_svg

    return draughts, draughts_svg


@app.cell
def _(mo):
    board_class_choice = mo.ui.radio(
        options=["Board (10x10 international)", "AmericanBoard (8x8)"],
        value="Board (10x10 international)",
        label="Board class",
    )
    board_class_choice
    return (board_class_choice,)


@app.cell
def _(board_class_choice, draughts):
    # Pick the right class + a sensible default FEN/move. svg_max is the
    # viewBox extent for that board (490 for 10x10, 400 for 8x8) so the
    # arrow xy sliders span the right pixel range.
    if "AmericanBoard" in board_class_choice.value:
        board_cls = draughts.AmericanBoard
        default_fen = draughts.AmericanBoard().fen
        default_move = "22-18"
        svg_max = 400
    else:
        board_cls = draughts.Board
        default_fen = draughts.Board().fen
        default_move = "34-29"
        svg_max = 490
    return board_cls, default_fen, default_move, svg_max


@app.cell
def _(default_fen, default_move, mo):
    fen_input = mo.ui.text_area(
        value=default_fen,
        label="FEN",
        full_width=True,
    )
    move_input = mo.ui.text(
        value=default_move,
        label="Move",
    )
    mo.vstack([fen_input, move_input])
    return fen_input, move_input


@app.cell
def _(board_cls, draughts_svg, move_input):
    # Compute the natural pixel xy of the move's tail/head squares, used as
    # initial slider values. Falls back to viewBox center if parsing fails.
    from draughts import Color as _Color

    def _split_endpoints(move_str: str):
        _parts = move_str.replace("x", "-").split("-")
        return int(_parts[0]), int(_parts[-1])

    _board_size = board_cls().shape[0]  # 10 or 8 — `shape` is instance-only

    try:
        _tail_sq, _head_sq = _split_endpoints(move_input.value)
        default_tail_xy = draughts_svg._get_square_center(
            _tail_sq - 1, _board_size, _Color.WHITE, 20
        )
        default_head_xy = draughts_svg._get_square_center(
            _head_sq - 1, _board_size, _Color.WHITE, 20
        )
    except Exception:
        _center = (_board_size * 45 + 40) / 2
        default_tail_xy = (_center, _center)
        default_head_xy = (_center, _center)
    return default_head_xy, default_tail_xy


@app.cell
def _(default_head_xy, default_tail_xy, mo, svg_max):
    # Four sliders, pixel-coords in the SVG viewBox. Sliders re-default
    # whenever the move text changes (the cell's deps include the parsed
    # move's xy). Drag freely between resets.
    tail_x = mo.ui.slider(
        0, svg_max, value=int(default_tail_xy[0]), step=5, label="tail x"
    )
    tail_y = mo.ui.slider(
        0, svg_max, value=int(default_tail_xy[1]), step=5, label="tail y"
    )
    head_x = mo.ui.slider(
        0, svg_max, value=int(default_head_xy[0]), step=5, label="head x"
    )
    head_y = mo.ui.slider(
        0, svg_max, value=int(default_head_xy[1]), step=5, label="head y"
    )
    mo.vstack(
        [
            mo.md("**Arrow xy** (drag to reposition):"),
            mo.hstack([tail_x, tail_y, head_x, head_y], justify="start"),
        ]
    )
    return head_x, head_y, tail_x, tail_y


@app.cell
def _(
    board_cls,
    draughts_svg,
    fen_input,
    head_x,
    head_y,
    mo,
    move_input,
    tail_x,
    tail_y,
):
    # Render before SVG. Render after SVG with `lastmove` highlight only
    # (no native Arrow — we inject a custom slider-controlled arrow below).
    # Both error paths (bad FEN, illegal move) keep the other panel alive.

    _ARROW_TEMPLATE = (
        '<defs>'
        '<marker id="custArrow" viewBox="0 0 10 10" refX="9" refY="5"'
        ' markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" fill="green"/>'
        '</marker></defs>'
        '<line x1="{tx}" y1="{ty}" x2="{hx}" y2="{hy}"'
        ' stroke="green" stroke-width="5" opacity="0.85"'
        ' marker-end="url(#custArrow)"/>'
    )

    # `min-height` on the sub keeps both cards' SVGs aligned at the same Y
    # — long FENs wrap to ~3 lines, short captions like "move: 34-29" are
    # padded to match. Long custom FENs that exceed min-height will still
    # push their card's SVG down (acceptable edge case).
    _CARD_TEMPLATE = (
        '<div style="border:1px solid #d0d0d0;border-radius:10px;'
        'padding:14px;background:#fafafa;display:inline-block;'
        'box-shadow:0 1px 3px rgba(0,0,0,.04);">'
        '<div style="font-weight:600;margin-bottom:4px;">{title}</div>'
        '<div style="font-family:ui-monospace,monospace;font-size:11px;'
        'color:#666;margin-bottom:10px;max-width:360px;min-height:48px;'
        'word-break:break-all;">{sub}</div>'
        '<div>{body}</div></div>'
    )

    def _card(title, sub, body):
        return mo.Html(_CARD_TEMPLATE.format(title=title, sub=sub, body=body))

    try:
        before_board = board_cls.from_fen(fen_input.value)
        before_svg = str(draughts_svg.board(before_board, size=360))
        before_err = None
    except Exception as _e:
        before_board = None
        before_svg = ""
        before_err = "Bad FEN: " + str(_e)

    after_svg = ""
    after_err = None
    if before_board is not None:
        try:
            after_board = board_cls.from_fen(fen_input.value)
            after_board.push_uci(move_input.value)
            _raw = str(draughts_svg.board(after_board, size=360))
            _arrow = _ARROW_TEMPLATE.format(
                tx=tail_x.value,
                ty=tail_y.value,
                hx=head_x.value,
                hy=head_y.value,
            )
            after_svg = _raw.replace("</svg>", _arrow + "</svg>")
        except Exception as _e:
            after_err = "Move not playable: " + str(_e)

    _err_html = '<div style="color:#c33">{}</div>'

    before_card = _card(
        "PROMPT",
        "FEN: " + fen_input.value,
        before_svg if before_svg else _err_html.format(before_err),
    )
    after_card = _card(
        "LABEL",
        "move: " + move_input.value,
        after_svg if after_svg else _err_html.format(after_err or "n/a"),
    )
    mo.hstack([before_card, after_card], justify="start", gap=2)
    return


if __name__ == "__main__":
    app.run()
