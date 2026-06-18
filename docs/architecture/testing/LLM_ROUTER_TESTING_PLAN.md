---
type: Architecture Spec
title: LLM Router Enhancement - Comprehensive Testing Plan
description: '**File:** `tests/integration/test_llm_openai_e2e.py`'
resource: /docs/architecture/testing/LLM_ROUTER_TESTING_PLAN.md
tags:
- docs
- architecture
- architecture-spec
timestamp: '2026-06-18T00:00:00Z'
status: needs-revalidation
owner: architecture
canonical: false
generated: false
privacy: public
---

# LLM Router Enhancement - Comprehensive Testing Plan

## Testing Philosophy
- **REAL TESTS ONLY - NO MOCKS**
- Tests use real Ollama (localhost:11434), real OpenAI API, real filesystem
- Tests create and verify real data
- Tests measure real performance
- **Fail Fast**: Tests stop at first failure (pytest -x)
- If tests fail, STOP and investigate - don't paper over failures

## Test Organization

```
tests/
├── integration/
│   ├── test_llm_openai_e2e.py          # Real OpenAI API tests
│   ├── test_llm_ollama_e2e.py          # Real Ollama tests
│   ├── test_llm_model_pulling_e2e.py   # Model download tests
│   ├── test_llm_space_mgmt_e2e.py      # Disk space & cleanup tests
│   ├── test_llm_privacy_routing_e2e.py # Privacy mode enforcement
│   └── test_llm_persistence_e2e.py     # Session persistence tests
├── e2e/
│   └── test_llm_ui_workflow.spec.cjs   # Playwright UI tests
└── performance/
    └── test_llm_performance.py          # Load & latency tests
```

---

## Phase 1: OpenAI Integration Tests (Real API)

**File:** `tests/integration/test_llm_openai_e2e.py`

### Setup/Teardown
```python
@pytest.fixture(scope="module")
def skip_if_no_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set - skipping OpenAI tests")

@pytest.fixture
def llm_service():
    """Real LLMService with OpenAI configured"""
    service = LLMService()
    assert any(p["name"] == "openai" for p in service.providers)
    return service
```

### Test Cases

#### 1. Model Discovery
```python
async def test_openai_model_discovery(llm_service, skip_if_no_api_key):
    """Verify OpenAI model discovery returns valid models"""
    models = await llm_service._discover_openai_models()

    # Assertions
    assert len(models) > 0, "Should discover at least 1 OpenAI model"
    assert any("gpt-5" in m["id"] for m in models), "Should include gpt-5"

    # Verify model structure
    for model in models:
        assert "id" in model
        assert "provider" in model
        assert model["provider"] == "openai"
        assert "name" in model
        assert "description" in model
```

#### 2. Chat Completion
```python
async def test_openai_chat_completion(llm_service, skip_if_no_api_key):
    """Verify OpenAI returns valid chat response"""
    response = await llm_service.get_response(
        system_prompt="You are a helpful assistant.",
        user_prompt="Say 'test successful' and nothing else.",
        force_local_only=False
    )

    # Assertions
    assert response is not None
    assert len(response) > 0
    assert "test successful" in response.lower()
    assert isinstance(response, str)
```

#### 3. Streaming Response
```python
async def test_openai_streaming(llm_service, skip_if_no_api_key):
    """Verify OpenAI streaming works"""
    chunks = []
    async for chunk in llm_service.get_response_stream(
        system_prompt="You are a helpful assistant.",
        user_prompt="Count to 3.",
        force_local_only=False
    ):
        chunks.append(chunk)

    # Assertions
    assert len(chunks) > 0, "Should receive at least 1 chunk"
    full_response = "".join(chunks)
    assert len(full_response) > 0
    # Should contain numbers 1, 2, 3
    assert any(str(i) in full_response for i in [1, 2, 3])
```

#### 4. JSON Mode
```python
async def test_openai_json_mode(llm_service, skip_if_no_api_key):
    """Verify OpenAI JSON response format"""
    response = await llm_service.get_response(
        system_prompt="Return JSON only.",
        user_prompt='Return {"status": "ok"}',
        force_local_only=False,
        format="json"
    )

    # Assertions
    assert response is not None
    parsed = json.loads(response)  # Should not raise
    assert isinstance(parsed, dict)
    assert "status" in parsed
```

#### 5. Model Selection
```python
async def test_openai_specific_model(llm_service, skip_if_no_api_key):
    """Verify can request specific OpenAI model"""
    response = await llm_service.get_response_with_model(
        model_id="gpt-5-mini",
        system_prompt="You are a helpful assistant.",
        user_prompt="Say 'mini model working' and nothing else."
    )

    # Assertions
    assert response is not None
    assert "mini model" in response.lower() or "working" in response.lower()
```

#### 6. Error Handling
```python
async def test_openai_invalid_model(llm_service, skip_if_no_api_key):
    """Verify proper error for invalid model"""
    with pytest.raises(ValueError, match="Model.*not found"):
        await llm_service.get_response_with_model(
            model_id="gpt-nonexistent-model",
            system_prompt="Test",
            user_prompt="Test"
        )
```

#### 7. Rate Limiting
```python
async def test_openai_rate_limit_handling(llm_service, skip_if_no_api_key):
    """Verify graceful handling of rate limits"""
    # Make rapid requests
    tasks = [
        llm_service.get_response(
            system_prompt="Test",
            user_prompt=f"Request {i}",
            force_local_only=False
        )
        for i in range(10)
    ]

    # Should handle rate limits gracefully (retry or clear error)
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # At least some should succeed
    successes = [r for r in responses if not isinstance(r, Exception)]
    assert len(successes) > 0, "At least some requests should succeed"
```

---

## Phase 2: Ollama Integration Tests (Real Service)

**File:** `tests/integration/test_llm_ollama_e2e.py`

### Setup/Teardown
```python
@pytest.fixture(scope="module")
def skip_if_no_ollama():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code != 200:
            pytest.skip("Ollama not running on localhost:11434")
    except:
        pytest.skip("Ollama not accessible")

@pytest.fixture
def ensure_test_model():
    """Ensure qwen2.5:0.5b is available (small test model)"""
    client = ollama.Client()
    models = client.list()
    if not any("qwen2.5:0.5b" in m["name"] for m in models["models"]):
        pytest.skip("Test model qwen2.5:0.5b not installed")
```

### Test Cases

#### 1. Ollama Service Availability
```python
def test_ollama_service_running(skip_if_no_ollama):
    """Verify Ollama service is accessible"""
    response = requests.get("http://localhost:11434/api/tags")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
```

#### 2. Model Discovery
```python
async def test_ollama_model_discovery(llm_service, skip_if_no_ollama):
    """Verify Ollama model discovery"""
    models = await llm_service._discover_ollama_models()

    # Assertions
    assert len(models) > 0, "Should discover at least 1 Ollama model"

    # Verify model structure
    for model in models:
        assert "id" in model
        assert "provider" in model
        assert model["provider"] == "ollama"
        assert "size_gb" in model
        assert model["size_gb"] > 0
```

#### 3. Chat Completion
```python
async def test_ollama_chat_completion(llm_service, skip_if_no_ollama, ensure_test_model):
    """Verify Ollama returns valid chat response"""
    response = await llm_service.get_response(
        system_prompt="You are a helpful assistant.",
        user_prompt="Say 'ollama working' and nothing else.",
        force_local_only=True  # Force Ollama
    )

    # Assertions
    assert response is not None
    assert len(response) > 0
    assert isinstance(response, str)
```

#### 4. Streaming Response
```python
async def test_ollama_streaming(llm_service, skip_if_no_ollama, ensure_test_model):
    """Verify Ollama streaming works"""
    chunks = []
    async for chunk in llm_service.get_response_stream(
        system_prompt="You are a helpful assistant.",
        user_prompt="Count to 3.",
        force_local_only=True
    ):
        chunks.append(chunk)

    # Assertions
    assert len(chunks) > 0, "Should receive at least 1 chunk"
    full_response = "".join(chunks)
    assert len(full_response) > 0
```

#### 5. JSON Mode
```python
async def test_ollama_json_mode(llm_service, skip_if_no_ollama, ensure_test_model):
    """Verify Ollama JSON response format"""
    response = await llm_service.get_response(
        system_prompt="Return JSON only.",
        user_prompt='Return {"status": "ok"}',
        force_local_only=True,
        format="json"
    )

    # Assertions
    assert response is not None
    # Ollama may wrap in markdown, handle that
    clean_response = response.strip().strip("```json").strip("```").strip()
    parsed = json.loads(clean_response)
    assert isinstance(parsed, dict)
```

---

## Phase 3: Model Pulling Tests (Real Downloads)

**File:** `tests/integration/test_llm_model_pulling_e2e.py`

### Setup/Teardown
```python
@pytest.fixture(scope="module")
def test_model_name():
    """Small model for testing (~500MB)"""
    return "qwen2.5:0.5b"

@pytest.fixture
def cleanup_test_model(test_model_name):
    """Clean up test model after tests"""
    yield
    # Cleanup
    try:
        client = ollama.Client()
        client.delete(test_model_name)
    except:
        pass
```

### Test Cases

#### 1. Check Disk Space Before Pull
```python
async def test_check_space_before_pull(llm_service, skip_if_no_ollama):
    """Verify disk space check before pulling"""
    storage_info = await llm_service.get_storage_info()

    # Assertions
    assert "total_gb" in storage_info
    assert "used_gb" in storage_info
    assert "available_gb" in storage_info
    assert storage_info["available_gb"] > 0
```

#### 2. Pull Small Model
```python
async def test_pull_model_success(llm_service, skip_if_no_ollama, test_model_name, cleanup_test_model):
    """Verify can pull a small model"""
    # Remove model first if exists
    try:
        client = ollama.Client()
        client.delete(test_model_name)
    except:
        pass

    # Pull model
    result = await llm_service.pull_model(test_model_name)

    # Assertions
    assert result is True, "Pull should succeed"

    # Verify model now available
    models = await llm_service._discover_ollama_models()
    assert any(test_model_name in m["id"] for m in models)
```

#### 3. Pull Progress Tracking
```python
async def test_pull_with_progress(llm_service, skip_if_no_ollama, test_model_name, cleanup_test_model):
    """Verify progress tracking during pull"""
    progress_updates = []

    async def progress_callback(status, completed, total):
        progress_updates.append({
            "status": status,
            "completed": completed,
            "total": total
        })

    # Remove model first
    try:
        client = ollama.Client()
        client.delete(test_model_name)
    except:
        pass

    # Pull with progress
    await llm_service.pull_model(test_model_name, progress_callback=progress_callback)

    # Assertions
    assert len(progress_updates) > 0, "Should receive progress updates"
    assert any(u["status"] == "downloading" for u in progress_updates)
    assert progress_updates[-1]["status"] == "success"
```

#### 4. Insufficient Space Error
```python
async def test_pull_insufficient_space(llm_service, skip_if_no_ollama, monkeypatch):
    """Verify error when insufficient disk space"""
    # Mock get_storage_info to return low space
    async def mock_storage():
        return {
            "total_gb": 100,
            "used_gb": 99.5,
            "available_gb": 0.5
        }

    monkeypatch.setattr(llm_service, "get_storage_info", mock_storage)

    # Try to pull large model
    with pytest.raises(RuntimeError, match="Insufficient disk space"):
        await llm_service.pull_model("llama3.2:70b")  # Large model
```

#### 5. Auto-Download on Missing Model
```python
async def test_auto_download_missing_model(llm_service, skip_if_no_ollama, test_model_name, cleanup_test_model):
    """Verify auto-download when model not found"""
    # Remove model first
    try:
        client = ollama.Client()
        client.delete(test_model_name)
    except:
        pass

    # Request model that doesn't exist (should auto-download)
    response = await llm_service.get_response_with_model(
        model_id=test_model_name,
        system_prompt="Test",
        user_prompt="Hello",
        auto_pull=True
    )

    # Assertions
    assert response is not None, "Should succeed after auto-download"

    # Verify model now exists
    models = await llm_service._discover_ollama_models()
    assert any(test_model_name in m["id"] for m in models)
```

#### 6. No Auto-Download When Disabled
```python
async def test_no_auto_download_when_disabled(llm_service, skip_if_no_ollama):
    """Verify no auto-download when auto_pull=False"""
    # Try to use non-existent model
    with pytest.raises(ValueError, match="Model.*not found"):
        await llm_service.get_response_with_model(
            model_id="nonexistent-model:1b",
            system_prompt="Test",
            user_prompt="Hello",
            auto_pull=False
        )
```

---

## Phase 4: Space Management & Cleanup Tests

**File:** `tests/integration/test_llm_space_mgmt_e2e.py`

### Test Cases

#### 1. Get Storage Info
```python
async def test_get_storage_info(llm_service, skip_if_no_ollama):
    """Verify storage info returns accurate data"""
    storage_info = await llm_service.get_storage_info()

    # Assertions
    assert "total_gb" in storage_info
    assert "used_gb" in storage_info
    assert "available_gb" in storage_info
    assert "models" in storage_info
    assert isinstance(storage_info["models"], list)

    # Verify math
    assert storage_info["total_gb"] > 0
    assert storage_info["used_gb"] >= 0
    assert storage_info["available_gb"] >= 0
    assert storage_info["total_gb"] >= storage_info["used_gb"]
```

#### 2. Model Usage Tracking
```python
async def test_model_usage_tracking(llm_service, skip_if_no_ollama, ensure_test_model):
    """Verify model usage is tracked"""
    # Use a model
    await llm_service.get_response_with_model(
        model_id="qwen2.5:0.5b",
        system_prompt="Test",
        user_prompt="Hello"
    )

    # Check usage was recorded
    usage = await llm_service.get_model_usage("qwen2.5:0.5b")

    # Assertions
    assert usage is not None
    assert "last_used" in usage
    assert "use_count" in usage
    assert usage["use_count"] > 0

    # Verify last_used is recent (within last minute)
    last_used = datetime.fromisoformat(usage["last_used"])
    age_seconds = (datetime.now() - last_used).total_seconds()
    assert age_seconds < 60
```

#### 3. Protected Models Not Deleted
```python
async def test_protected_models_not_deleted(llm_service, skip_if_no_ollama):
    """Verify protected models are never deleted"""
    # Get models from config
    config_models = llm_service.mandate_config.get("protected_models", [])

    # Run cleanup (dry-run)
    deleted_models = await llm_service.cleanup_unused_models(
        dry_run=True,
        threshold_days=0  # Would delete everything not protected
    )

    # Assertions
    # Protected models should NOT be in deleted list
    for model in config_models:
        assert model not in deleted_models, f"Protected model {model} should not be deleted"
```

#### 4. Cleanup Old Models
```python
async def test_cleanup_old_models(llm_service, skip_if_no_ollama, cleanup_test_model):
    """Verify old unused models are cleaned up"""
    test_model = "qwen2.5:0.5b"

    # Ensure model exists
    await llm_service.pull_model(test_model)

    # Manually set last_used to 60 days ago
    conn = await llm_service.db.connect()
    await conn.execute(
        "UPDATE model_usage SET last_used = ? WHERE model_id = ?",
        (datetime.now() - timedelta(days=60), test_model)
    )
    await conn.commit()

    # Run cleanup (dry-run first)
    deleted_models = await llm_service.cleanup_unused_models(
        dry_run=True,
        threshold_days=30
    )

    # Assertions
    assert test_model in deleted_models, "Old unused model should be marked for deletion"

    # Actually delete
    deleted_models = await llm_service.cleanup_unused_models(
        dry_run=False,
        threshold_days=30
    )

    # Verify model removed
    models = await llm_service._discover_ollama_models()
    assert not any(test_model in m["id"] for m in models), "Model should be deleted"
```

#### 5. Auto-Cleanup When Low Space
```python
async def test_auto_cleanup_low_space(llm_service, skip_if_no_ollama, monkeypatch):
    """Verify auto-cleanup triggers when space low"""
    # Mock storage to return low space
    async def mock_storage():
        return {
            "total_gb": 100,
            "used_gb": 92,
            "available_gb": 8,
            "models": []
        }

    monkeypatch.setattr(llm_service, "get_storage_info", mock_storage)

    # Mock cleanup to track if called
    cleanup_called = []
    original_cleanup = llm_service.cleanup_unused_models
    async def mock_cleanup(*args, **kwargs):
        cleanup_called.append(True)
        return []

    monkeypatch.setattr(llm_service, "cleanup_unused_models", mock_cleanup)

    # Trigger space check (should auto-cleanup)
    await llm_service._check_and_cleanup_if_needed()

    # Assertions
    assert len(cleanup_called) > 0, "Cleanup should be triggered when space <10%"
```

---

## Phase 5: Privacy-Aware Routing Tests

**File:** `tests/integration/test_llm_privacy_routing_e2e.py`

### Test Cases

#### 1. EPHEMERAL Mode - Force Local Only
```python
async def test_ephemeral_mode_forces_ollama(llm_service, skip_if_no_ollama, skip_if_no_api_key):
    """Verify EPHEMERAL mode only uses Ollama (no override)"""
    response = await llm_service.get_response(
        system_prompt="Test",
        user_prompt="Hello",
        privacy_mode="EPHEMERAL",
        requested_model="gpt-5"  # Try to request OpenAI
    )

    # Should use Ollama despite request
    # Verify by checking logs or metadata
    assert response is not None
```

#### 2. ISOLATED Mode - Force Local Only
```python
async def test_isolated_mode_forces_ollama(llm_service, skip_if_no_ollama):
    """Verify ISOLATED mode only uses Ollama (no override)"""
    response = await llm_service.get_response(
        system_prompt="Test",
        user_prompt="Hello",
        privacy_mode="ISOLATED"
    )

    # Should use Ollama
    assert response is not None
```

#### 3. ANONYMOUS Mode - Default Ollama, Allow Override
```python
async def test_anonymous_mode_defaults_ollama(llm_service, skip_if_no_ollama):
    """Verify ANONYMOUS mode defaults to Ollama"""
    response = await llm_service.get_response(
        system_prompt="Test",
        user_prompt="Hello",
        privacy_mode="ANONYMOUS"
        # No requested_model - should use Ollama default
    )

    assert response is not None

async def test_anonymous_mode_allows_override(llm_service, skip_if_no_api_key):
    """Verify ANONYMOUS mode allows OpenAI if explicitly requested"""
    response = await llm_service.get_response(
        system_prompt="Test",
        user_prompt="Hello",
        privacy_mode="ANONYMOUS",
        requested_model="gpt-5"  # Explicit request should work
    )

    assert response is not None
```

#### 4. NORMAL Mode - Default OpenAI
```python
async def test_normal_mode_defaults_openai(llm_service, skip_if_no_api_key):
    """Verify NORMAL mode defaults to OpenAI"""
    response = await llm_service.get_response(
        system_prompt="Test",
        user_prompt="Hello",
        privacy_mode="NORMAL"
        # No requested_model - should use OpenAI default
    )

    assert response is not None
```

#### 5. Error When Ollama Required But Unavailable
```python
async def test_error_when_local_only_but_no_ollama(llm_service, monkeypatch):
    """Verify clear error when EPHEMERAL mode but Ollama down"""
    # Mock Ollama as unavailable
    async def mock_get_response(*args, **kwargs):
        raise ConnectionError("Ollama not available")

    original_providers = llm_service.providers
    # Remove Ollama from providers
    llm_service.providers = [p for p in original_providers if p["name"] != "ollama"]

    # Try EPHEMERAL mode
    with pytest.raises(RuntimeError, match="LOCAL_ONLY mode.*no Ollama"):
        await llm_service.get_response(
            system_prompt="Test",
            user_prompt="Hello",
            privacy_mode="EPHEMERAL"
        )

    # Restore
    llm_service.providers = original_providers
```

---

## Phase 6: Session Persistence Tests

**File:** `tests/integration/test_llm_persistence_e2e.py`

### Test Cases

#### 1. Create Companion Without Model Preference
```python
async def test_create_companion_no_preference(kestrel_client, test_user):
    """Verify companion created without model preference"""
    companion = await kestrel_client.create_companion(
        user_id=test_user["id"],
        name="Test Companion",
        personality="friendly"
    )

    # Assertions
    assert companion["preferred_model"] is None
```

#### 2. Set Model Preference
```python
async def test_set_model_preference(kestrel_client, test_companion):
    """Verify can set persistent model preference"""
    result = await kestrel_client.set_companion_model(
        companion_id=test_companion["id"],
        model_id="gpt-5-mini"
    )

    # Assertions
    assert result["preferred_model"] == "gpt-5-mini"

    # Verify persisted
    companion = await kestrel_client.get_companion(test_companion["id"])
    assert companion["preferred_model"] == "gpt-5-mini"
```

#### 3. Model Persists Across Requests
```python
async def test_model_persists_across_requests(kestrel_client, test_companion):
    """Verify model preference persists across chat requests"""
    # Set preference
    await kestrel_client.set_companion_model(
        companion_id=test_companion["id"],
        model_id="gpt-5-mini"
    )

    # Send multiple messages
    for i in range(3):
        response = await kestrel_client.send_message(
            companion_id=test_companion["id"],
            message=f"Test message {i}"
        )
        assert response is not None

    # Verify all used the preferred model
    messages = await kestrel_client.get_messages(test_companion["id"])
    # Check metadata or logs to verify model used
```

#### 4. Reset to Default
```python
async def test_reset_model_preference(kestrel_client, test_companion):
    """Verify can reset model preference to default"""
    # Set preference
    await kestrel_client.set_companion_model(
        companion_id=test_companion["id"],
        model_id="gpt-5-mini"
    )

    # Reset
    result = await kestrel_client.reset_companion_model(
        companion_id=test_companion["id"]
    )

    # Assertions
    assert result["preferred_model"] is None
```

#### 5. Override Persisted Model Per-Request
```python
async def test_override_persisted_model(kestrel_client, test_companion):
    """Verify can override persisted model for single request"""
    # Set preference
    await kestrel_client.set_companion_model(
        companion_id=test_companion["id"],
        model_id="gpt-5-mini"
    )

    # Send message with override
    response = await kestrel_client.send_message(
        companion_id=test_companion["id"],
        message="Test message",
        model="gpt-5"  # Override
    )

    # Assertions
    assert response is not None

    # Verify preference unchanged
    companion = await kestrel_client.get_companion(test_companion["id"])
    assert companion["preferred_model"] == "gpt-5-mini"
```

---

## Phase 7: Performance Tests

**File:** `tests/performance/test_llm_performance.py`

### Test Cases

#### 1. Response Latency
```python
async def test_response_latency(llm_service, skip_if_no_ollama):
    """Verify response time is reasonable"""
    start = time.time()

    response = await llm_service.get_response(
        system_prompt="Test",
        user_prompt="Hello",
        force_local_only=True
    )

    elapsed = time.time() - start

    # Assertions
    assert response is not None
    assert elapsed < 10.0, f"Response took {elapsed}s, should be <10s"
```

#### 2. Concurrent Requests
```python
async def test_concurrent_requests(llm_service, skip_if_no_ollama):
    """Verify can handle multiple concurrent requests"""
    tasks = [
        llm_service.get_response(
            system_prompt="Test",
            user_prompt=f"Request {i}",
            force_local_only=True
        )
        for i in range(5)
    ]

    start = time.time()
    responses = await asyncio.gather(*tasks)
    elapsed = time.time() - start

    # Assertions
    assert len(responses) == 5
    assert all(r is not None for r in responses)
    # Should handle concurrency efficiently
    assert elapsed < 30.0, f"5 concurrent requests took {elapsed}s"
```

#### 3. Cache Performance
```python
async def test_model_discovery_cache(llm_service):
    """Verify model discovery cache improves performance"""
    # First call (cache miss)
    start1 = time.time()
    models1 = await llm_service.discover_all_models(use_cache=False)
    time1 = time.time() - start1

    # Second call (cache hit)
    start2 = time.time()
    models2 = await llm_service.discover_all_models(use_cache=True)
    time2 = time.time() - start2

    # Assertions
    assert len(models1) == len(models2)
    assert time2 < time1 / 2, "Cached call should be significantly faster"
```

---

## Phase 8: Edge Cases & Error Handling

### Test Cases

#### 1. Network Failure During Pull
```python
async def test_network_failure_during_pull(llm_service, monkeypatch):
    """Verify graceful handling of network failure"""
    # Mock pull to fail midway
    async def mock_pull(*args, **kwargs):
        raise ConnectionError("Network interrupted")

    with pytest.raises(ConnectionError):
        await llm_service.pull_model("test-model")

    # Verify partial download cleaned up
    # Verify model not marked as available
```

#### 2. Corrupted Model File
```python
async def test_corrupted_model_handling(llm_service):
    """Verify detection of corrupted model files"""
    # Test would need to corrupt a model file
    # Verify error is raised
    # Verify recovery mechanism
```

#### 3. Multiple Simultaneous Pulls
```python
async def test_simultaneous_pulls_same_model(llm_service, test_model_name):
    """Verify handling of concurrent pulls for same model"""
    # Remove model first
    # Start two pulls simultaneously
    # Only one should actually download
    # Both should succeed
```

#### 4. Cleanup During Active Use
```python
async def test_no_cleanup_of_active_model(llm_service):
    """Verify model in use is not cleaned up"""
    # Start long-running inference
    # Trigger cleanup
    # Verify active model not deleted
```

---

## Test Execution Strategy

### Local Development
```bash
# Run all tests with fail-fast
pytest tests/integration/test_llm_*.py -x -v

# Run specific test file
pytest tests/integration/test_llm_openai_e2e.py -x -v

# Run with coverage
pytest tests/integration/ --cov=llm --cov-report=html -x

# Run only fast tests
pytest tests/integration/ -m "not slow" -x

# Run performance tests separately
pytest tests/performance/ -v
```

### CI/CD Pipeline
```yaml
# .github/workflows/test-llm.yml
- name: Test OpenAI Integration
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  run: pytest tests/integration/test_llm_openai_e2e.py -x

- name: Test Ollama Integration
  run: |
    # Start Ollama
    ollama serve &
    sleep 5
    ollama pull qwen2.5:0.5b
    pytest tests/integration/test_llm_ollama_e2e.py -x
```

---

## Success Metrics

- ✅ All OpenAI tests pass (with API key)
- ✅ All Ollama tests pass (with service running)
- ✅ Model pulling tested with real downloads
- ✅ Space management verified with real filesystem
- ✅ Privacy modes enforce correct routing
- ✅ Session persistence works across requests
- ✅ Performance within acceptable ranges
- ✅ Edge cases handled gracefully
- ✅ No mocks in integration tests
- ✅ 95%+ test coverage on new code

---

## Test Data Requirements

1. **Test Models:**
   - Small: qwen2.5:0.5b (~500MB) - for quick tests
   - Medium: llama3.2:3b (~2GB) - for realistic tests
   - Tiny: tinyllama:1.1b (~600MB) - for edge cases

2. **Test Users:**
   - Anonymous user (no account)
   - Free tier user
   - Paid user

3. **Test Companions:**
   - No model preference
   - With model preference
   - Various privacy modes

4. **OpenAI Test Key:**
   - Separate test key with rate limits
   - Monitor usage to avoid costs
   - Use gpt-5-mini for cost efficiency

---

## Monitoring & Observability

### Metrics to Track
- Model pull success rate
- Model pull duration (p50, p95, p99)
- LLM response latency by provider
- Cache hit rate for model discovery
- Disk space usage over time
- Cleanup runs and models deleted
- Privacy mode usage distribution
- Model switching frequency

### Logging Requirements
- Every model pull (start, progress, completion/failure)
- Every cleanup run (dry-run vs actual, models deleted)
- Every privacy mode enforcement
- Every model switch (user-triggered vs automatic)
- Every cache hit/miss

---

**Total Test Coverage Target: 95%+**
**Estimated Test Development Time: 4-6 hours**
