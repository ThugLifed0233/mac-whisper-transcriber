"""Spawn-safe Streamlit entry point."""

from __future__ import annotations

import runpy


def main() -> None:
    """Execute the page only in Streamlit's primary script process."""

    runpy.run_module(
        "mac_whisper_transcriber.ui",
        run_name="__streamlit_ui__",
    )


if __name__ == "__main__":
    main()
