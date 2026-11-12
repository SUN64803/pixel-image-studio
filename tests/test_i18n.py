"""i18n coverage tests.

The UI is authored in Thai and translated at runtime by ``studio/i18n.js``:
static HTML text nodes/attributes and literal ``tr("…")`` calls in
``studio/app.js`` are looked up in the ``EN`` dictionary. A missing key used to
blank the UI (``I18N.tr`` returned ``null``), so these tests keep the dictionary
in sync and guard the fallback that makes a missing key harmless.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
I18N_JS = ROOT / "studio" / "i18n.js"
APP_JS = ROOT / "studio" / "app.js"
INDEX_HTML = ROOT / "studio" / "index.html"

THAI = re.compile(r"[\u0E00-\u0E7F]")

# Language names are shown in their own language in the language picker.
ALLOWED_UNTRANSLATED = {"ไทย (Thai)"}


def _load_en() -> dict[str, str]:
    src = I18N_JS.read_text(encoding="utf-8")
    match = re.search(r"const EN = \{(.*?)\n  \};", src, re.S)
    assert match, "EN dictionary not found in studio/i18n.js"
    body = match.group(1)
    body = re.sub(r"^\s*//.*$", "", body, flags=re.M)  # section comments
    body = body.rstrip().rstrip(",")  # trailing comma before the closing brace
    return json.loads("{" + body + "}")


class _TextAndAttrs(HTMLParser):
    """Collect text nodes and translatable attributes, skipping data-no-i18n."""

    ATTRS = ("placeholder", "title", "aria-label")

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.texts: list[str] = []
        self.attrs: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if self._skip_depth:
            self._skip_depth += 1
            return
        if any(k == "data-no-i18n" for k, _ in attrs):
            self._skip_depth = 1
            return
        for key, value in attrs:
            if key in self.ATTRS and value:
                self.attrs.append(value)

    def handle_startendtag(self, tag, attrs):
        if not self._skip_depth:
            for key, value in attrs:
                if key in self.ATTRS and value:
                    self.attrs.append(value)

    def handle_endtag(self, tag):
        if self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth:
            self.texts.append(data)


def _norm(text: str) -> str:
    return " ".join(text.split())


def test_en_dictionary_is_not_empty_and_has_no_blank_values():
    en = _load_en()
    assert len(en) > 400
    blank = [key for key, value in en.items() if not str(value).strip()]
    assert not blank, f"blank translations: {blank[:5]}"


def test_every_html_string_is_translated():
    en = _load_en()
    parser = _TextAndAttrs()
    parser.feed(INDEX_HTML.read_text(encoding="utf-8"))
    candidates = [_norm(t) for t in parser.texts] + [_norm(a) for a in parser.attrs]
    missing = sorted(
        {
            text
            for text in candidates
            if THAI.search(text)
            and text not in en
            and text not in ALLOWED_UNTRANSLATED
        }
    )
    assert not missing, "Thai strings missing from the EN dictionary: " + repr(missing[:8])


def test_literal_tr_calls_are_translated():
    en = _load_en()
    src = APP_JS.read_text(encoding="utf-8")
    literals = re.findall(r'\btr\(\s*"((?:[^"\\]|\\.)*)"\s*\)', src)
    assert literals, "expected literal tr(\"…\") calls in studio/app.js"
    missing = []
    for raw in literals:
        text = raw.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")
        if text.strip() not in en:
            missing.append(text.strip())
    assert not missing, f"literal tr() calls without an EN entry: {missing}"


def test_public_tr_never_returns_null():
    src = I18N_JS.read_text(encoding="utf-8")
    # The public wrapper must fall back to the source string; the internal
    # translateOne() is allowed to return null, the exported one is not.
    assert "function tr(text) {" in src
    assert 'return text == null ? "" : String(text);' in src
    assert "window.I18N = {" in src and "tr," in src
    # app.js additionally guards against a missing/broken bridge.
    app = APP_JS.read_text(encoding="utf-8")
    assert "const tr = (s) => (window.I18N && I18N.tr(s)) || s;" in app
