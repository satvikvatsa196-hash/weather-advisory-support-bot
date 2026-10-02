import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock
from langchain_core.messages import AIMessage
from backend.main import app

@pytest.fixture
def anyio_backend():
    return 'asyncio'

@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@pytest.mark.asyncio
async def test_valid_chat_request():
    mock_result = {
        "messages": [AIMessage(content="You should definitely stay inside today.")],
        "location": "Bhopal",
        "matched_sops": [{"sop": {"id": "sop_extreme_rain_system"}}]
    }
    with patch("backend.main.graph.ainvoke", new_callable=AsyncMock) as mock_graph:
        mock_graph.return_value = mock_result
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/chat", json={"session_id": "test-session", "message": "Is it good for cycling in Bhopal?"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session"
        assert data["response"] == "You should definitely stay inside today."
        assert data["location"] == "Bhopal"
        assert data["matched_sops"] == ["sop_extreme_rain_system"]
        
        # Verify session_id propagation
        mock_graph.assert_called_once()
        config_passed = mock_graph.call_args[1].get('config')
        assert config_passed["configurable"]["thread_id"] == "test-session"

@pytest.mark.asyncio
async def test_empty_message():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/chat", json={"session_id": "test-session", "message": ""})
    assert response.status_code == 422 # FastAPI validation error

@pytest.mark.asyncio
async def test_empty_session_id():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/chat", json={"session_id": "", "message": "hello"})
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_graph_error():
    with patch("backend.main.graph.ainvoke", new_callable=AsyncMock) as mock_graph:
        mock_graph.side_effect = Exception("Internal graph error")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/chat", json={"session_id": "test", "message": "hello"})
        
        assert response.status_code == 500
        data = response.json()
        assert data["detail"] == "Internal server error"
        # Ensure raw exception is not leaked

@pytest.mark.asyncio
async def test_separate_sessions():
    with patch("backend.main.graph.ainvoke", new_callable=AsyncMock) as mock_graph:
        mock_graph.return_value = {"messages": [AIMessage(content="Response")]}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            await ac.post("/chat", json={"session_id": "session1", "message": "hello"})
            await ac.post("/chat", json={"session_id": "session2", "message": "hello"})
        
        assert mock_graph.call_count == 2
        config1 = mock_graph.call_args_list[0][1]['config']
        config2 = mock_graph.call_args_list[1][1]['config']
        assert config1["configurable"]["thread_id"] == "session1"
        assert config2["configurable"]["thread_id"] == "session2"
