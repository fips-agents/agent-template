"""Tests for fipsagents.server.graph_store."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fipsagents.server.graph_store import (
    NullGraphStore,
    _build_property_string,
    _escape_cypher_value,
    _parse_agtype,
    _validate_label,
    create_age_graph_store,
)

try:
    import asyncpg  # noqa: F401
    from fipsagents.server.graph_store import AgeGraphStore, initialise_age_schema
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

age_only = pytest.mark.skipif(not HAS_ASYNCPG, reason="asyncpg not installed")


def _mock_pool(fetch_return=None):
    """Build a MagicMock asyncpg pool with async-context acquire support."""
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=fetch_return or [])
    conn.fetchrow = AsyncMock(return_value=None)
    conn.execute = AsyncMock()
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    pool.close = AsyncMock()
    return pool, conn


def _make_record(**kwargs):
    """Simulate an asyncpg Record with .values() support."""
    record = MagicMock()
    record.values.return_value = list(kwargs.values())
    return record


def _make_store(pool=None, graph_name="test_graph"):
    if not HAS_ASYNCPG:
        pytest.skip("asyncpg not installed")
    if pool is None:
        pool, _ = _mock_pool()
    return AgeGraphStore(pool, graph_name)


class TestValidateLabel:
    @pytest.mark.parametrize("label", ["Entity", "my_label", "A123", "_priv"])
    def test_valid(self, label):
        _validate_label(label)

    @pytest.mark.parametrize("label", ["123abc", "my-label", "", "has space"])
    def test_invalid(self, label):
        with pytest.raises(ValueError, match="invalid label"):
            _validate_label(label)


class TestEscapeCypherValue:
    @pytest.mark.parametrize("val, expected", [
        ("hello", '"hello"'),
        (42, "42"),
        (3.14, "3.14"),
        (True, "true"),
        (False, "false"),
        (None, "null"),
    ])
    def test_supported_types(self, val, expected):
        assert _escape_cypher_value(val) == expected

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="unsupported Cypher value type"):
            _escape_cypher_value([1, 2])


class TestBuildPropertyString:
    @pytest.mark.parametrize("props", [None, {}])
    def test_empty(self, props):
        assert _build_property_string(props) == ""

    def test_with_values(self):
        assert _build_property_string({"name": "test", "count": 5}) == ' {name: "test", count: 5}'


class TestParseAgtype:
    def test_vertex_suffix_stripped(self):
        assert _parse_agtype('{"id": 123, "label": "Entity"}::vertex') == {"id": 123, "label": "Entity"}

    def test_plain_json(self):
        assert _parse_agtype("42") == 42

    def test_none_passthrough(self):
        assert _parse_agtype(None) is None

    def test_non_json_falls_back_to_string(self):
        assert _parse_agtype("not-json") == "not-json"


class TestNullGraphStore:
    @pytest.mark.asyncio
    async def test_add_node_returns_zero(self):
        assert await NullGraphStore().add_node("Entity") == 0

    @pytest.mark.asyncio
    async def test_add_edge_returns_zero(self):
        assert await NullGraphStore().add_edge(1, 2, "REL") == 0

    @pytest.mark.asyncio
    async def test_get_node_returns_none(self):
        assert await NullGraphStore().get_node(1) is None

    @pytest.mark.asyncio
    async def test_get_neighbors_returns_empty(self):
        assert await NullGraphStore().get_neighbors(1) == []

    @pytest.mark.asyncio
    async def test_query_cypher_returns_empty(self):
        assert await NullGraphStore().query_cypher("MATCH (n) RETURN n") == []

    @pytest.mark.asyncio
    async def test_search_nodes_returns_empty(self):
        assert await NullGraphStore().search_nodes("Entity") == []

    @pytest.mark.asyncio
    async def test_delete_node_returns_false(self):
        assert await NullGraphStore().delete_node(1) is False

    @pytest.mark.asyncio
    async def test_delete_edge_returns_false(self):
        assert await NullGraphStore().delete_edge(1) is False

    @pytest.mark.asyncio
    async def test_close_is_noop(self):
        await NullGraphStore().close()


@age_only
class TestAgeGraphStoreAddNode:
    @pytest.mark.asyncio
    async def test_returns_id(self):
        pool, conn = _mock_pool(fetch_return=[_make_record(id="844424930131969")])
        assert await _make_store(pool=pool).add_node("Entity") == 844424930131969

    @pytest.mark.asyncio
    async def test_with_properties(self):
        pool, conn = _mock_pool(fetch_return=[_make_record(id="123")])
        await _make_store(pool=pool).add_node("Entity", {"name": "test"})
        assert 'name: "test"' in conn.fetch.await_args.args[0]

    @pytest.mark.asyncio
    async def test_db_failure_returns_zero(self):
        pool, conn = _mock_pool()
        conn.fetch = AsyncMock(side_effect=RuntimeError("connection lost"))
        assert await _make_store(pool=pool).add_node("Entity") == 0


@age_only
class TestAgeGraphStoreAddEdge:
    @pytest.mark.asyncio
    async def test_returns_id(self):
        pool, conn = _mock_pool(fetch_return=[_make_record(id="456")])
        assert await _make_store(pool=pool).add_edge(1, 2, "KNOWS") == 456

    @pytest.mark.asyncio
    async def test_no_rows_returns_zero(self):
        pool, _ = _mock_pool(fetch_return=[])
        assert await _make_store(pool=pool).add_edge(1, 2, "KNOWS") == 0

    @pytest.mark.asyncio
    async def test_db_failure_returns_zero(self):
        pool, conn = _mock_pool()
        conn.fetch = AsyncMock(side_effect=RuntimeError("boom"))
        assert await _make_store(pool=pool).add_edge(1, 2, "KNOWS") == 0


@age_only
class TestAgeGraphStoreGetNode:
    @pytest.mark.asyncio
    async def test_found(self):
        pool, _ = _mock_pool(fetch_return=[
            _make_record(id="123", lbl='"Entity"', props='{"name": "test"}'),
        ])
        node = await _make_store(pool=pool).get_node(123)
        assert node is not None
        assert node["id"] == 123
        assert node["label"] == "Entity"
        assert node["properties"] == {"name": "test"}

    @pytest.mark.asyncio
    async def test_not_found(self):
        pool, _ = _mock_pool(fetch_return=[])
        assert await _make_store(pool=pool).get_node(999) is None

    @pytest.mark.asyncio
    async def test_db_failure(self):
        pool, conn = _mock_pool()
        conn.fetch = AsyncMock(side_effect=RuntimeError("timeout"))
        assert await _make_store(pool=pool).get_node(1) is None


@age_only
class TestAgeGraphStoreGetNeighbors:
    @pytest.mark.asyncio
    async def test_returns_list(self):
        pool, _ = _mock_pool(fetch_return=[_make_record(
            m_id="10", m_lbl='"Person"', m_props='{"name": "Alice"}',
            e_id="20", e_lbl='"KNOWS"', e_props="{}",
        )])
        neighbors = await _make_store(pool=pool).get_neighbors(1)
        assert len(neighbors) == 1
        assert neighbors[0]["id"] == 10
        assert neighbors[0]["label"] == "Person"
        assert neighbors[0]["edge_id"] == 20
        assert neighbors[0]["edge_label"] == "KNOWS"

    @pytest.mark.asyncio
    async def test_direction_out(self):
        pool, conn = _mock_pool(fetch_return=[])
        await _make_store(pool=pool).get_neighbors(1, direction="out")
        assert "-[e]->" in conn.fetch.await_args.args[0]

    @pytest.mark.asyncio
    async def test_direction_in(self):
        pool, conn = _mock_pool(fetch_return=[])
        await _make_store(pool=pool).get_neighbors(1, direction="in")
        assert "<-[e]-" in conn.fetch.await_args.args[0]

    @pytest.mark.asyncio
    async def test_with_edge_label(self):
        pool, conn = _mock_pool(fetch_return=[])
        await _make_store(pool=pool).get_neighbors(1, edge_label="KNOWS")
        assert "[e:KNOWS]" in conn.fetch.await_args.args[0]

    @pytest.mark.asyncio
    async def test_db_failure_returns_empty(self):
        pool, conn = _mock_pool()
        conn.fetch = AsyncMock(side_effect=RuntimeError("down"))
        assert await _make_store(pool=pool).get_neighbors(1) == []


@age_only
class TestAgeGraphStoreQueryCypher:
    @pytest.mark.asyncio
    async def test_returns_results(self):
        pool, _ = _mock_pool(fetch_return=[_make_record(result='"hello"'), _make_record(result="42")])
        rows = await _make_store(pool=pool).query_cypher("MATCH (n) RETURN n")
        assert rows == [{"result": "hello"}, {"result": 42}]

    @pytest.mark.asyncio
    async def test_empty(self):
        pool, _ = _mock_pool(fetch_return=[])
        assert await _make_store(pool=pool).query_cypher("MATCH (n) RETURN n") == []

    @pytest.mark.asyncio
    async def test_db_failure(self):
        pool, conn = _mock_pool()
        conn.fetch = AsyncMock(side_effect=RuntimeError("sql error"))
        assert await _make_store(pool=pool).query_cypher("MATCH (n) RETURN n") == []

    @pytest.mark.asyncio
    async def test_warns_on_params(self, caplog):
        pool, _ = _mock_pool(fetch_return=[])
        with caplog.at_level("WARNING"):
            await _make_store(pool=pool).query_cypher("MATCH (n) RETURN n", params={"x": 1})
        assert "params" in caplog.text.lower()


@age_only
class TestAgeGraphStoreSearchNodes:
    @pytest.mark.asyncio
    async def test_by_label(self):
        pool, conn = _mock_pool(fetch_return=[_make_record(id="1", lbl='"Entity"', props="{}")])
        results = await _make_store(pool=pool).search_nodes("Entity")
        assert len(results) == 1
        assert "Entity" in conn.fetch.await_args.args[0]

    @pytest.mark.asyncio
    async def test_with_filter(self):
        pool, conn = _mock_pool(fetch_return=[])
        await _make_store(pool=pool).search_nodes("Entity", property_filter={"name": "test"})
        sql = conn.fetch.await_args.args[0]
        assert "WHERE" in sql and 'n.name = "test"' in sql

    @pytest.mark.asyncio
    async def test_db_failure(self):
        pool, conn = _mock_pool()
        conn.fetch = AsyncMock(side_effect=RuntimeError("boom"))
        assert await _make_store(pool=pool).search_nodes("Entity") == []


@age_only
class TestAgeGraphStoreDelete:
    @pytest.mark.asyncio
    async def test_delete_node_found(self):
        pool, _ = _mock_pool(fetch_return=[_make_record(deleted="true")])
        assert await _make_store(pool=pool).delete_node(1) is True

    @pytest.mark.asyncio
    async def test_delete_node_not_found(self):
        pool, _ = _mock_pool(fetch_return=[])
        assert await _make_store(pool=pool).delete_node(999) is False

    @pytest.mark.asyncio
    async def test_delete_node_db_failure(self):
        pool, conn = _mock_pool()
        conn.fetch = AsyncMock(side_effect=RuntimeError("err"))
        assert await _make_store(pool=pool).delete_node(1) is False

    @pytest.mark.asyncio
    async def test_delete_edge_found(self):
        pool, _ = _mock_pool(fetch_return=[_make_record(deleted="true")])
        assert await _make_store(pool=pool).delete_edge(1) is True

    @pytest.mark.asyncio
    async def test_delete_edge_not_found(self):
        pool, _ = _mock_pool(fetch_return=[])
        assert await _make_store(pool=pool).delete_edge(999) is False

    @pytest.mark.asyncio
    async def test_delete_edge_db_failure(self):
        pool, conn = _mock_pool()
        conn.fetch = AsyncMock(side_effect=RuntimeError("err"))
        assert await _make_store(pool=pool).delete_edge(1) is False


@age_only
class TestAgeGraphStoreClose:
    @pytest.mark.asyncio
    async def test_calls_pool_close(self):
        pool, _ = _mock_pool()
        await _make_store(pool=pool).close()
        pool.close.assert_awaited_once()


@age_only
class TestInitialiseAgeSchema:
    @pytest.mark.asyncio
    async def test_creates_extension_and_graph(self):
        pool, conn = _mock_pool()
        conn.fetchrow = AsyncMock(return_value=None)
        await initialise_age_schema(pool, "test_graph")
        calls = [c.args[0] for c in conn.execute.await_args_list]
        assert any("CREATE EXTENSION" in c for c in calls)
        assert any("create_graph" in c for c in calls)

    @pytest.mark.asyncio
    async def test_skips_graph_creation_if_exists(self):
        pool, conn = _mock_pool()
        conn.fetchrow = AsyncMock(return_value={"exists": 1})
        await initialise_age_schema(pool, "test_graph")
        calls = [c.args[0] for c in conn.execute.await_args_list]
        assert not any("create_graph" in c for c in calls)

    @pytest.mark.asyncio
    async def test_raises_on_invalid_graph_name(self):
        pool, _ = _mock_pool()
        with pytest.raises(ValueError, match="invalid graph_name"):
            await initialise_age_schema(pool, "bad-name")


@age_only
class TestCreateAgeGraphStore:
    @pytest.mark.asyncio
    async def test_empty_database_url_returns_null(self):
        assert isinstance(await create_age_graph_store(database_url=""), NullGraphStore)

    @pytest.mark.asyncio
    async def test_invalid_graph_name_returns_null(self):
        store = await create_age_graph_store(database_url="postgresql://localhost/x", graph_name="bad-name")
        assert isinstance(store, NullGraphStore)

    @pytest.mark.asyncio
    async def test_pool_failure_returns_null(self):
        with patch("asyncpg.create_pool", new=AsyncMock(side_effect=RuntimeError("refused"))):
            store = await create_age_graph_store(database_url="postgresql://localhost/x")
        assert isinstance(store, NullGraphStore)

    @pytest.mark.asyncio
    async def test_schema_failure_returns_null_and_closes_pool(self):
        pool, _ = _mock_pool()
        with (
            patch("asyncpg.create_pool", new=AsyncMock(return_value=pool)),
            patch("fipsagents.server.graph_store.initialise_age_schema",
                  new=AsyncMock(side_effect=RuntimeError("schema fail"))),
        ):
            store = await create_age_graph_store(database_url="postgresql://localhost/x")
        assert isinstance(store, NullGraphStore)
        pool.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_success_returns_age_store(self):
        pool, conn = _mock_pool()
        conn.fetchrow = AsyncMock(return_value=None)
        with patch("asyncpg.create_pool", new=AsyncMock(return_value=pool)):
            store = await create_age_graph_store(database_url="postgresql://localhost/x", graph_name="test_graph")
        assert isinstance(store, AgeGraphStore)
