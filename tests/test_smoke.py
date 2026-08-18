"""Package-installability smoke test.

CONTRACT: `import emorec` must succeed from any working directory once the
package is installed (`pip install -e ".[dev]"`) — this is what proves the
src-layout packaging is wired correctly.
"""


def test_import_emorec() -> None:
    import emorec  # noqa: F401
