import pytest

from mcp_server.db import is_safe_select

SAFE = [
    "SELECT 1",
    "select id, name from users where guild_id = 123",
    "  SELECT * FROM events ORDER BY created_at DESC  ",
    "WITH recent AS (SELECT * FROM events) SELECT * FROM recent",
    "SELECT count(*) FROM tokens;",  # single trailing semicolon allowed
    "SELECT 1 /* note a/b and 2*3 /* nested */ end */",
    "SELECT id FROM users WHERE bio = 'I do create things; delete later'",
    "SELECT * FROM t WHERE note = 'it''s set'",
]

UNSAFE = [
    ("", "empty"),
    ("INSERT INTO users VALUES (1)", "insert"),
    ("UPDATE users SET name = 'x'", "update"),
    ("DELETE FROM users", "delete"),
    ("DROP TABLE users", "drop"),
    ("ALTER TABLE users ADD COLUMN x int", "alter"),
    ("TRUNCATE users", "truncate"),
    ("GRANT SELECT ON users TO bob", "grant"),
    ("SELECT 1; DROP TABLE users", "multiple statements"),
    ("SELECT 1; SELECT 2", "multiple statements"),
    ("SELECT 1 /* c */; SELECT 2", "multiple statements"),
    ("WITH x AS (DELETE FROM users RETURNING *) SELECT * FROM x", "delete"),
    ("COPY users TO '/tmp/x'", "copy"),
    ("CALL do_thing()", "call"),
    ("EXPLAIN ANALYZE DELETE FROM users", "not a select"),
    ("SELECT 1 -- ok\nDELETE FROM users", "delete after line comment"),
    ("SELECT * FROM t WHERE a = 'ok'; DROP TABLE t", "multiple statements"),
]


@pytest.mark.parametrize("sql", SAFE)
def test_safe_queries_pass(sql):
    ok, reason = is_safe_select(sql)
    assert ok, reason


@pytest.mark.parametrize("sql,_label", UNSAFE)
def test_unsafe_queries_rejected(sql, _label):
    ok, _ = is_safe_select(sql)
    assert not ok
