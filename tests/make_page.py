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


#: A page of genuinely wrapped prose. The statement page above is hard-wrapped
#: into short lines the model returns separately; this one is the case that
#: needs the line pass, because the markdown prompt reflows each paragraph into
#: a single run of text under a single box.
PROSE = [
    "The Committee has reviewed the submitted materials and finds that the "
    "applicant has satisfied each of the conditions enumerated in Section 4 "
    "of the governing ordinance. Notice was published in a newspaper of "
    "general circulation not less than fourteen days before the hearing, and "
    "abutters within three hundred feet received written notice by certified "
    "mail. No objection was received from any abutter, and no member of the "
    "public appeared in opposition at the hearing held on the twelfth of May.",
    "Accordingly, the variance is granted subject to the condition that "
    "construction commence within twelve months of the date of this decision. "
    "Any person aggrieved by this decision may appeal to the Superior Court "
    "within twenty days of the filing of this decision with the Town Clerk.",
]

PROSE_WIDTH = 1400


def wrap(draw, text: str, fnt, max_width: int) -> list[str]:
    """Break text where the page will break it, so tests know the true lines."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if current and draw.textlength(trial, font=fnt) > max_width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def prose_lines() -> list[str]:
    """The printed lines of the prose page, in order."""
    draw = ImageDraw.Draw(Image.new("RGB", (1700, 2200), "white"))
    body = font(34)
    return [line for para in PROSE for line in wrap(draw, para, body, PROSE_WIDTH)]


def build_prose_image(path: Path) -> Path:
    image = Image.new("RGB", (1700, 2200), "white")
    draw = ImageDraw.Draw(image)
    draw.text((150, 140), "TOWN OF NORTHFIELD", fill="black", font=font(52))
    draw.text((150, 230), "Zoning Board of Appeals - Decision", fill="black", font=font(38))

    body = font(34)
    y = 360
    for para in PROSE:
        for line in wrap(draw, para, body, PROSE_WIDTH):
            draw.text((150, y), line, fill="black", font=body)
            y += 52
        y += 40

    image.save(path, dpi=(200, 200))
    return path


def build_prose_pdf(path: Path) -> Path:
    png = path.with_suffix(".png")
    build_prose_image(png)
    with Image.open(png) as image:
        image.convert("RGB").save(path, "PDF", resolution=200.0)
    return path


def build_page_pdf(path: Path) -> Path:
    png = path.with_suffix(".png")
    build_page_image(png)
    with Image.open(png) as image:
        image.convert("RGB").save(path, "PDF", resolution=200.0)
    return path
