# tests/make_page.py
"""Build the synthetic statement page used by integration tests.

The footer at 86% page height is the point of this fixture: DeepSeek-OCR-2
drops it on a full-page pass and reads it correctly from a crop.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FOOTER_PHONE = "1-800-555-0147"
_FONT = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"

BODY = [
    "Account Holder: Matthew Zagaja",
    "Account Number: 4471-9920-3318",
    "Statement Period: 01 April 2026 to 30 June 2026",
    "",
    "This statement summarizes all transactions posted to your",
    "account during the period noted above. Please review each",
    "entry carefully and report any discrepancy within 30 days.",
    "",
    "Opening Balance ................................ $ 12,480.55",
    "Deposits and Credits ........................... $  3,201.10",
    "Withdrawals and Debits ......................... $  1,795.62",
    "Interest Earned ................................ $     44.18",
    "Closing Balance ................................ $ 13,930.21",
]


def font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(_FONT, size)
    except OSError:
        return ImageFont.load_default(size)


def build_page_image(path: Path) -> Path:
    image = Image.new("RGB", (1700, 2200), "white")
    draw = ImageDraw.Draw(image)
    draw.text((150, 140), "ACME CORPORATION", fill="black", font=font(56))
    draw.text((150, 240), "Quarterly Statement of Account", fill="black", font=font(38))

    y = 380
    for line in BODY:
        if line:
            draw.text((150, y), line, fill="black", font=font(32))
        y += 58

    draw.text(
        (150, 1900),
        f"Questions? Call {FOOTER_PHONE} or visit acme.example.com",
        fill="black",
        font=font(28),
    )
    image.save(path, dpi=(200, 200))
    return path


def build_page_pdf(path: Path) -> Path:
    png = path.with_suffix(".png")
    build_page_image(png)
    with Image.open(png) as image:
        image.convert("RGB").save(path, "PDF", resolution=200.0)
    return path
