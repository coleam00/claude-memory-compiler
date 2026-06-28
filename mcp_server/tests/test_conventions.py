from mcp_server import conventions


def test_get_conventions_returns_agents_md():
    text = conventions.get_conventions()
    assert "Repository Guidelines" in text or "Project Structure" in text


def test_get_mistakes_full():
    text = conventions.get_mistakes()
    assert "Mistake" in text


def test_get_mistakes_topic_filter():
    # The mistakes file documents a Modal/deferReply rule.
    text = conventions.get_mistakes("modal")
    assert "modal" in text.lower()
    # Filtering should shrink the output vs the full file.
    assert len(text) < len(conventions.get_mistakes())
