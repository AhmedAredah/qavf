"""Reference tables retain field structure for arbitrary plugin defaults."""

from pydantic import BaseModel, Field

from qavf.docgen import params_table


def test_parameter_defaults_and_aliases_are_single_line_table_cells():
    class Parameters(BaseModel):
        pattern: str = "a|b"
        text: str = "first\nsecond\r\t"
        quoted: str = Field("`literal``", alias="alias|`name")

    rows = params_table(Parameters).splitlines()
    assert len(rows) == 5
    assert "`a\\|b`" in rows[2]
    assert "`first\\nsecond\\r\\t`" in rows[3]
    assert "``` `literal`` ```" in rows[4]
    assert "``alias\\|`name``" in rows[4]
    for row in rows:
        # Table separators exclude escaped pipes inside values.
        assert len(__import__("re").findall(r"(?<!\\)\|", row)) == 5
