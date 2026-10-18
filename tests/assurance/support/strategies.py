"""Domain-shaped generators retain adversarial cases without rejection filtering."""

from hypothesis import strategies as st

NULLABLE_NUMBER = st.one_of(
    st.none(),
    st.just(float("nan")),
    st.just(-999.0),
    st.integers(-12, 12).map(float),
)
NULLABLE_TEXT = st.one_of(st.none(), st.sampled_from(["", " ", "\t", "NA", "a", "b", "c"]))
KEY_ROW = st.tuples(
    st.one_of(st.none(), st.just(-999), st.integers(-4, 4)),
    st.one_of(st.none(), st.sampled_from(["", " ", "NA", "a", "b", "c"])),
)
KEY_ROWS = st.lists(KEY_ROW, max_size=18)
