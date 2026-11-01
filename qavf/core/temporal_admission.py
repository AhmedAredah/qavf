"""Preserve Arrow nanosecond instants before DuckDB's microsecond timezone bridge."""

from __future__ import annotations

from qavf.sources.base import SourceError


class TemporalAdmission:
    """Normalize explicitly governed instants without copying timestamp buffers."""

    def __init__(self, spec, canonical=()):
        self.spec = spec
        self.canonical = {name.casefold(): name for name in canonical}

    def schema(self, schema):
        import pyarrow as pa

        fields, notes = [], {}
        for field in schema:
            dtype = field.type
            if self.contains_zoned_nanoseconds(dtype):
                if not pa.types.is_timestamp(dtype):
                    raise SourceError(
                        f"nested timezone-aware nanosecond column {field.name!r} cannot be "
                        "preserved by DuckDB; flatten and declare its timestamp columns first"
                    )
                target = self.spec.rename.get(
                    field.name, self.canonical.get(field.name.casefold(), field.name)
                )
                column = (
                    self.spec.schema_spec.columns.get(target) if self.spec.schema_spec else None
                )
                if column is None or column.dtype != "datetime" or not column.timezone:
                    raise SourceError(
                        f"timezone-aware nanosecond column {field.name!r} would lose precision "
                        "in DuckDB; declare dtype='datetime' and timezone, "
                        "then select datetime_unit"
                    )
                fields.append(field.with_type(pa.timestamp("ns")))
                notes[field.name] = {
                    "source_type": str(dtype),
                    "canonical_type": "TIMESTAMP_NS",
                    "canonical_timezone": "UTC",
                    "requested_unit": column.datetime_unit,
                }
            else:
                fields.append(field)
        return pa.schema(fields, metadata=schema.metadata), notes

    @staticmethod
    def contains_zoned_nanoseconds(dtype):
        import pyarrow as pa

        if pa.types.is_timestamp(dtype):
            return dtype.unit == "ns" and dtype.tz is not None
        if pa.types.is_dictionary(dtype):
            return TemporalAdmission.contains_zoned_nanoseconds(dtype.value_type)
        if isinstance(dtype, pa.ExtensionType):
            return TemporalAdmission.contains_zoned_nanoseconds(dtype.storage_type)
        return any(
            TemporalAdmission.contains_zoned_nanoseconds(dtype.field(index).type)
            for index in range(dtype.num_fields)
        )

    def table(self, table):
        schema, notes = self.schema(table.schema)
        return (table.cast(schema, safe=True) if notes else table), notes

    def eager(self, value, budget=None):
        module = type(value).__module__.split(".")[0]
        if module == "pandas":
            # Ordinary pandas frames keep the existing direct registration path.
            if not any(
                getattr(dtype, "unit", None) == "ns" and getattr(dtype, "tz", None) is not None
                for dtype in value.dtypes
            ):
                return value, {}
            import pyarrow as pa

            value = pa.Table.from_pandas(value, preserve_index=False)
        elif module == "polars":
            import polars as pl

            if isinstance(value, pl.LazyFrame):
                empty = pl.DataFrame(schema=value.collect_schema()).to_arrow()
                _, notes = self.schema(empty.schema)
                if not notes:
                    return value, {}
                value = (value.limit(budget + 1) if budget is not None else value).collect()
            value = value.to_arrow()
        elif module != "pyarrow":
            return value, {}
        import pyarrow as pa

        if isinstance(value, pa.RecordBatch):
            value = pa.Table.from_batches([value])
        if isinstance(value, pa.Table):
            return self.table(value)
        return value, {}
