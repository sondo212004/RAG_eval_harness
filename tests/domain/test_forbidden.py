from rag_eval_harness.domain.forbidden import find_forbidden, parse_needles


# AC11
def test_detects_forbidden_string_case_insensitive():
    files = {"a.py": "x = 1\nname = 'AcmeCorp'\n", "b.md": "sạch"}

    assert find_forbidden(files, ["acmecorp"]) == [("a.py", 2, "acmecorp")]


def test_parse_needles_skips_comments_and_blanks():
    assert parse_needles("# comment\n\nAcme\n  beta  \n") == ["acme", "beta"]
