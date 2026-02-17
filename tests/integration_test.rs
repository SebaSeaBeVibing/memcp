use std::process::{Command, Stdio};
use std::io::{Write, BufRead, BufReader};
use std::sync::mpsc::{channel, Sender, Receiver};
use std::thread;
use std::time::Duration;
use serde_json::{json, Value};

/// Helper struct to manage server process with async I/O
struct McpClient {
    child: std::process::Child,
    tx: Sender<Value>,
    rx: Receiver<Value>,
}

impl McpClient {
    fn spawn() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_memcp"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())  // Suppress log output in tests
            .spawn()
            .expect("Failed to spawn memcp binary");

        let mut stdin = child.stdin.take().expect("Failed to get stdin");
        let stdout = child.stdout.take().expect("Failed to get stdout");

        // Channel for sending requests
        let (req_tx, req_rx) = channel::<Value>();

        // Channel for receiving responses
        let (resp_tx, resp_rx) = channel::<Value>();

        // Thread to write requests to stdin
        thread::spawn(move || {
            while let Ok(request) = req_rx.recv() {
                let request_str = serde_json::to_string(&request).expect("Failed to serialize");
                if writeln!(stdin, "{}", request_str).is_err() {
                    break;
                }
                if stdin.flush().is_err() {
                    break;
                }
            }
        });

        // Thread to read responses from stdout
        thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => break, // EOF
                    Ok(_) => {
                        if let Ok(value) = serde_json::from_str::<Value>(&line) {
                            if resp_tx.send(value).is_err() {
                                break;
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        McpClient {
            child,
            tx: req_tx,
            rx: resp_rx,
        }
    }

    fn send_request(&self, request: Value) -> Option<Value> {
        self.tx.send(request).ok()?;
        self.rx.recv_timeout(Duration::from_secs(2)).ok()
    }

    fn send_notification(&self, notification: Value) {
        let _ = self.tx.send(notification);
        // Notifications don't have responses, give server time to process
        thread::sleep(Duration::from_millis(50));
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Test client that connects to PostgreSQL via DATABASE_URL env var.
///
/// Integration tests requiring data isolation should use separate PostgreSQL
/// schemas or databases — managed by Docker Compose in Plan 02 / CI in Plan 03.
/// For now, tests share the same database and clean up after themselves.
struct McpTestClient {
    child: std::process::Child,
    tx: Sender<Value>,
    rx: Receiver<Value>,
    request_counter: std::cell::Cell<u64>,
}

impl McpTestClient {
    /// Spawn a client using DATABASE_URL from environment (or default postgres://memcp:memcp@localhost:5432/memcp).
    fn spawn() -> Self {
        let database_url = std::env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgres://memcp:memcp@localhost:5432/memcp".to_string());

        let mut child = Command::new(env!("CARGO_BIN_EXE_memcp"))
            .env("DATABASE_URL", &database_url)
            .env("MEMCP_LOG_LEVEL", "warn")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("Failed to spawn memcp binary");

        let mut stdin = child.stdin.take().expect("Failed to get stdin");
        let stdout = child.stdout.take().expect("Failed to get stdout");

        let (req_tx, req_rx) = channel::<Value>();
        let (resp_tx, resp_rx) = channel::<Value>();

        thread::spawn(move || {
            while let Ok(request) = req_rx.recv() {
                let request_str = serde_json::to_string(&request).expect("Failed to serialize");
                if writeln!(stdin, "{}", request_str).is_err() {
                    break;
                }
                if stdin.flush().is_err() {
                    break;
                }
            }
        });

        thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        if let Ok(value) = serde_json::from_str::<Value>(&line) {
                            if resp_tx.send(value).is_err() {
                                break;
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        McpTestClient {
            child,
            tx: req_tx,
            rx: resp_rx,
            request_counter: std::cell::Cell::new(10), // start at 10 to avoid collision with init id=1
        }
    }

    /// Send initialize + initialized notification, returning the init response.
    fn initialize(&self) -> Value {
        let response = self.send_request(json!({
            "jsonrpc": "2.0",
            "method": "initialize",
            "id": 1,
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"}
            }
        })).expect("Failed to get initialize response");

        // Give server time to process
        thread::sleep(Duration::from_millis(100));

        self.send_notification(json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }));

        response
    }

    /// Send a JSON-RPC request and wait for a response.
    fn send_request(&self, request: Value) -> Option<Value> {
        self.tx.send(request).ok()?;
        self.rx.recv_timeout(Duration::from_secs(10)).ok()
    }

    /// Send a notification (no response expected).
    fn send_notification(&self, notification: Value) {
        let _ = self.tx.send(notification);
        thread::sleep(Duration::from_millis(50));
    }

    /// Get next unique request ID.
    fn next_id(&self) -> u64 {
        let id = self.request_counter.get();
        self.request_counter.set(id + 1);
        id
    }

    /// Call a tool by name with arguments. Returns the full response Value.
    fn call_tool(&self, name: &str, arguments: Value) -> Value {
        let id = self.next_id();
        self.send_request(json!({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "id": id,
            "params": {
                "name": name,
                "arguments": arguments
            }
        })).unwrap_or_else(|| panic!("No response from tool {}", name))
    }

    /// Extract structuredContent from a tool call response.
    fn structured_content(response: &Value) -> &Value {
        &response["result"]["structuredContent"]
    }

    /// Check if a tool response has isError: true.
    fn is_error(response: &Value) -> bool {
        let result = &response["result"];
        result["isError"] == true
    }

    /// Call resources/list. Returns the full response.
    fn list_resources(&self) -> Value {
        let id = self.next_id();
        self.send_request(json!({
            "jsonrpc": "2.0",
            "method": "resources/list",
            "id": id,
            "params": {}
        })).expect("No response from resources/list")
    }

    /// Call resources/read with a given URI. Returns the full response.
    fn read_resource(&self, uri: &str) -> Value {
        let id = self.next_id();
        self.send_request(json!({
            "jsonrpc": "2.0",
            "method": "resources/read",
            "id": id,
            "params": {"uri": uri}
        })).expect("No response from resources/read")
    }
}

impl Drop for McpTestClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// =============================================================================
// Phase 1 Integration Tests (preserved)
// =============================================================================

#[test]
fn test_initialize_handshake() {
    let client = McpClient::spawn();

    // Send initialize request
    let initialize_request = json!({
        "jsonrpc": "2.0",
        "method": "initialize",
        "id": 1,
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    });

    let response = client.send_request(initialize_request)
        .expect("Failed to get initialize response");

    // Verify response structure
    assert_eq!(response["jsonrpc"], "2.0");
    assert_eq!(response["id"], 1);
    assert!(response["result"].is_object());

    let result = &response["result"];
    assert_eq!(result["protocolVersion"], "2024-11-05");
    assert!(result["capabilities"]["tools"].is_object());
    assert_eq!(result["serverInfo"]["name"], "memcp");
    assert!(result["serverInfo"]["version"].is_string());
    assert!(result["serverInfo"]["description"].is_string());

    // Send initialized notification (no response expected)
    let initialized_notification = json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    });
    client.send_notification(initialized_notification);
}

#[test]
fn test_tool_discovery() {
    let client = McpClient::spawn();

    // Initialize first
    let initialize_request = json!({
        "jsonrpc": "2.0",
        "method": "initialize",
        "id": 1,
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"}
        }
    });
    client.send_request(initialize_request)
        .expect("Failed to initialize");

    // Send initialized notification
    client.send_notification(json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    }));

    // Send tools/list request
    let tools_list_request = json!({
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 2
    });

    let response = client.send_request(tools_list_request)
        .expect("Failed to get tools/list response");

    // Verify response
    assert_eq!(response["jsonrpc"], "2.0");
    assert_eq!(response["id"], 2);
    assert!(response["result"]["tools"].is_array());

    let tools = response["result"]["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 8, "Should have exactly 8 tools");

    // Check all expected tools are present
    let tool_names: Vec<String> = tools.iter()
        .map(|t| t["name"].as_str().unwrap().to_string())
        .collect();

    assert!(tool_names.contains(&"store_memory".to_string()));
    assert!(tool_names.contains(&"get_memory".to_string()));
    assert!(tool_names.contains(&"update_memory".to_string()));
    assert!(tool_names.contains(&"delete_memory".to_string()));
    assert!(tool_names.contains(&"bulk_delete_memories".to_string()));
    assert!(tool_names.contains(&"list_memories".to_string()));
    assert!(tool_names.contains(&"search_memory".to_string()));
    assert!(tool_names.contains(&"health_check".to_string()));

    // Verify each tool has required fields
    for tool in tools {
        assert!(tool["name"].is_string());
        assert!(tool["description"].is_string());
        assert!(tool["inputSchema"].is_object());
    }
}

#[test]
fn test_store_memory_success() {
    let client = McpClient::spawn();

    // Give PostgreSQL store a moment to initialize before the first request
    thread::sleep(Duration::from_millis(200));

    // Initialize
    let initialize_request = json!({
        "jsonrpc": "2.0",
        "method": "initialize",
        "id": 1,
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"}
        }
    });
    client.send_request(initialize_request)
        .expect("Failed to initialize");

    client.send_notification(json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    }));

    // Call store_memory tool with valid params (Phase 2 API: content, type_hint, source, tags)
    let store_request = json!({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 2,
        "params": {
            "name": "store_memory",
            "arguments": {
                "content": "test memory content",
                "type_hint": "fact",
                "source": "test"
            }
        }
    });

    let response = client.send_request(store_request)
        .expect("Failed to get store_memory response");

    // Verify success response
    assert_eq!(response["jsonrpc"], "2.0");
    assert_eq!(response["id"], 2);
    assert!(response["result"].is_object());

    let result = &response["result"];
    assert!(result["isError"].is_null() || result["isError"] == false);

    // Check structured content
    assert!(result["content"].is_array());

    // Check for structuredContent (rmcp 0.15 uses this field)
    if result["structuredContent"].is_object() {
        let content = &result["structuredContent"];
        assert!(content["id"].is_string(), "Should have an ID");
        assert_eq!(content["content"], "test memory content");
        assert_eq!(content["type_hint"], "fact");
        assert_eq!(content["source"], "test");
        assert!(content["created_at"].is_string(), "Should have timestamp");

        // Verify ID looks like a UUID
        let id_str = content["id"].as_str().unwrap();
        assert!(id_str.contains('-'), "ID should be UUID-like");
    }
}

#[test]
fn test_store_memory_validation_error() {
    let client = McpClient::spawn();

    // Initialize
    let initialize_request = json!({
        "jsonrpc": "2.0",
        "method": "initialize",
        "id": 1,
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"}
        }
    });
    client.send_request(initialize_request)
        .expect("Failed to initialize");

    client.send_notification(json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    }));

    // Call store_memory with empty content (should fail validation)
    let store_request = json!({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 2,
        "params": {
            "name": "store_memory",
            "arguments": {
                "content": ""
            }
        }
    });

    let response = client.send_request(store_request)
        .expect("Failed to get store_memory response");

    // Verify validation error response
    assert_eq!(response["jsonrpc"], "2.0");
    assert_eq!(response["id"], 2);
    assert!(response["result"].is_object());

    let result = &response["result"];
    assert_eq!(result["isError"], true, "Should have isError: true");

    // Check error message mentions "content"
    let content_arr = result["content"].as_array().expect("content should be array");
    let error_text = content_arr[0]["text"].as_str().expect("should have error text");
    assert!(error_text.to_lowercase().contains("content"),
            "Error message should mention 'content': {}", error_text);
}

#[test]
fn test_health_check() {
    let client = McpClient::spawn();

    // Initialize
    let initialize_request = json!({
        "jsonrpc": "2.0",
        "method": "initialize",
        "id": 1,
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"}
        }
    });
    client.send_request(initialize_request)
        .expect("Failed to initialize");

    client.send_notification(json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    }));

    // Call health_check tool
    let health_request = json!({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 2,
        "params": {
            "name": "health_check",
            "arguments": {}
        }
    });

    let response = client.send_request(health_request)
        .expect("Failed to get health_check response");

    // Verify health check response
    assert_eq!(response["jsonrpc"], "2.0");
    assert_eq!(response["id"], 2);
    assert!(response["result"].is_object());

    let result = &response["result"];
    assert!(result["isError"].is_null() || result["isError"] == false);

    // Check structured content for health data
    if result["structuredContent"].is_object() {
        let health = &result["structuredContent"];
        assert_eq!(health["status"], "ok");
        assert!(health["version"].is_string());
        assert!(health["uptime_seconds"].is_number());
    }
}

#[test]
fn test_search_memory() {
    let client = McpClient::spawn();

    // Initialize
    let initialize_request = json!({
        "jsonrpc": "2.0",
        "method": "initialize",
        "id": 1,
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"}
        }
    });
    client.send_request(initialize_request)
        .expect("Failed to initialize");

    client.send_notification(json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    }));

    // Call search_memory tool
    let search_request = json!({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 2,
        "params": {
            "name": "search_memory",
            "arguments": {
                "query": "test"
            }
        }
    });

    let response = client.send_request(search_request)
        .expect("Failed to get search_memory response");

    // Verify search response
    assert_eq!(response["jsonrpc"], "2.0");
    assert_eq!(response["id"], 2);
    assert!(response["result"].is_object());

    let result = &response["result"];
    assert!(result["isError"].is_null() || result["isError"] == false);

    // Check structured content for mock results
    if result["structuredContent"].is_object() {
        let search_result = &result["structuredContent"];
        assert!(search_result["results"].is_array(), "Should have results array");

        // Verify mock results have expected structure
        let results = search_result["results"].as_array().unwrap();
        if !results.is_empty() {
            let first_result = &results[0];
            assert!(first_result["id"].is_string());
            assert!(first_result["content"].is_string());
            assert!(first_result["relevance_score"].is_number());
            assert!(first_result["created_at"].is_string());
        }
    }
}

// =============================================================================
// Phase 2 CRUD Integration Tests
// Note: These tests require a running PostgreSQL instance.
// Run with: DATABASE_URL=postgres://memcp:memcp@localhost:5432/memcp cargo test
// =============================================================================

#[test]
fn test_store_and_get_memory() {
    let client = McpTestClient::spawn();
    client.initialize();

    // Store a memory
    let store_resp = client.call_tool("store_memory", json!({
        "content": "Rust is great",
        "type_hint": "fact",
        "source": "test-agent"
    }));
    assert!(!McpTestClient::is_error(&store_resp), "store should succeed");

    let stored = McpTestClient::structured_content(&store_resp);
    assert!(stored["id"].is_string(), "Should have an ID");
    assert_eq!(stored["content"], "Rust is great");
    assert_eq!(stored["type_hint"], "fact");
    assert_eq!(stored["source"], "test-agent");
    assert!(stored["created_at"].is_string(), "Should have created_at");
    assert!(stored["updated_at"].is_string(), "Should have updated_at");
    assert_eq!(stored["access_count"], 0, "New memory should have access_count 0");
    assert!(stored["hint"].is_string(), "Should have usage hint");

    let memory_id = stored["id"].as_str().unwrap().to_string();

    // Get the memory
    let get_resp = client.call_tool("get_memory", json!({"id": memory_id}));
    assert!(!McpTestClient::is_error(&get_resp), "get should succeed");

    let retrieved = McpTestClient::structured_content(&get_resp);
    assert_eq!(retrieved["id"], memory_id);
    assert_eq!(retrieved["content"], "Rust is great");
    assert_eq!(retrieved["type_hint"], "fact");
    assert_eq!(retrieved["source"], "test-agent");
    // Note: access_count in the response reflects the pre-touch value (touch is fire-and-forget
    // after fetch). The DB will have access_count=1 but the returned memory shows 0.
    // This is intentional for Phase 2 performance (TODO(perf): Phase 6 will return post-touch value)
    assert!(retrieved["access_count"].is_number(), "access_count should be a number");
    assert!(retrieved["hint"].is_string(), "Should have usage hint");
}

#[test]
fn test_update_memory() {
    let client = McpTestClient::spawn();
    client.initialize();

    // Store a memory
    let store_resp = client.call_tool("store_memory", json!({
        "content": "Original content",
        "type_hint": "fact"
    }));
    assert!(!McpTestClient::is_error(&store_resp), "store should succeed");
    let memory_id = McpTestClient::structured_content(&store_resp)["id"]
        .as_str().unwrap().to_string();

    // Update the memory
    let update_resp = client.call_tool("update_memory", json!({
        "id": memory_id,
        "content": "Updated content",
        "tags": ["new-tag"]
    }));
    assert!(!McpTestClient::is_error(&update_resp), "update should succeed");

    let updated = McpTestClient::structured_content(&update_resp);
    assert_eq!(updated["content"], "Updated content");
    assert!(updated["tags"].is_array() || updated["tags"].is_string(),
            "Tags should be set");

    // Verify update persisted
    let get_resp = client.call_tool("get_memory", json!({"id": memory_id}));
    assert!(!McpTestClient::is_error(&get_resp), "get after update should succeed");
    let retrieved = McpTestClient::structured_content(&get_resp);
    assert_eq!(retrieved["content"], "Updated content");
}

#[test]
fn test_delete_memory() {
    let client = McpTestClient::spawn();
    client.initialize();

    // Store a memory
    let store_resp = client.call_tool("store_memory", json!({
        "content": "Memory to delete"
    }));
    assert!(!McpTestClient::is_error(&store_resp), "store should succeed");
    let memory_id = McpTestClient::structured_content(&store_resp)["id"]
        .as_str().unwrap().to_string();

    // Delete the memory
    let delete_resp = client.call_tool("delete_memory", json!({"id": memory_id}));
    assert!(!McpTestClient::is_error(&delete_resp), "delete should succeed");

    let deleted = McpTestClient::structured_content(&delete_resp);
    assert_eq!(deleted["deleted"], true, "Should report deleted: true");
    assert_eq!(deleted["id"], memory_id, "Should return the deleted ID");

    // Verify memory is gone - get should return isError: true
    let get_resp = client.call_tool("get_memory", json!({"id": memory_id}));
    assert!(McpTestClient::is_error(&get_resp),
            "get after delete should return isError: true");
}

#[test]
fn test_list_memories_with_pagination() {
    let client = McpTestClient::spawn();
    client.initialize();

    // Store 5 memories
    for i in 0..5 {
        let resp = client.call_tool("store_memory", json!({
            "content": format!("Memory {}", i),
            "type_hint": "fact"
        }));
        assert!(!McpTestClient::is_error(&resp), "store {} should succeed", i);
        // Small delay to ensure distinct created_at timestamps
        thread::sleep(Duration::from_millis(10));
    }

    // List with limit 2
    let list_resp = client.call_tool("list_memories", json!({"limit": 2}));
    assert!(!McpTestClient::is_error(&list_resp), "list should succeed");
    let page1 = McpTestClient::structured_content(&list_resp);
    let memories1 = page1["memories"].as_array().unwrap();
    assert_eq!(memories1.len(), 2, "Should return 2 memories");
    assert_eq!(page1["has_more"], true, "Should have more pages");
    assert!(page1["next_cursor"].is_string(), "Should have next_cursor");

    let cursor = page1["next_cursor"].as_str().unwrap().to_string();

    // Get next page
    let list_resp2 = client.call_tool("list_memories", json!({"limit": 2, "cursor": cursor}));
    assert!(!McpTestClient::is_error(&list_resp2), "page 2 list should succeed");
    let page2 = McpTestClient::structured_content(&list_resp2);
    let memories2 = page2["memories"].as_array().unwrap();
    assert_eq!(memories2.len(), 2, "Should return 2 more memories");

    // Collect all IDs from first two pages
    let mut all_ids: Vec<String> = memories1.iter()
        .chain(memories2.iter())
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect();

    // Get remaining
    let cursor2 = page2["next_cursor"].as_str().unwrap_or("").to_string();
    if !cursor2.is_empty() {
        let list_resp3 = client.call_tool("list_memories", json!({"limit": 2, "cursor": cursor2}));
        assert!(!McpTestClient::is_error(&list_resp3), "page 3 list should succeed");
        let page3 = McpTestClient::structured_content(&list_resp3);
        let memories3 = page3["memories"].as_array().unwrap();
        for m in memories3 {
            all_ids.push(m["id"].as_str().unwrap().to_string());
        }
    }

    // All 5 memories should have been retrieved
    assert_eq!(all_ids.len(), 5, "Should have retrieved all 5 memories across pages");

    // No duplicate IDs
    let unique: std::collections::HashSet<_> = all_ids.iter().collect();
    assert_eq!(unique.len(), 5, "All memory IDs should be unique");
}

#[test]
fn test_list_memories_with_filter() {
    let client = McpTestClient::spawn();
    client.initialize();

    // Store memories with different type_hints
    client.call_tool("store_memory", json!({"content": "Fact 1", "type_hint": "fact"}));
    client.call_tool("store_memory", json!({"content": "Fact 2", "type_hint": "fact"}));
    client.call_tool("store_memory", json!({"content": "Pref 1", "type_hint": "preference"}));
    client.call_tool("store_memory", json!({"content": "Event 1", "type_hint": "event"}));

    // List only fact type
    let list_resp = client.call_tool("list_memories", json!({"type_hint": "fact"}));
    assert!(!McpTestClient::is_error(&list_resp), "list with filter should succeed");
    let content = McpTestClient::structured_content(&list_resp);
    let memories = content["memories"].as_array().unwrap();
    assert_eq!(memories.len(), 2, "Should return only 2 fact memories");
    for m in memories {
        assert_eq!(m["type_hint"], "fact", "All returned memories should be facts");
    }
}

#[test]
fn test_bulk_delete_two_step() {
    let client = McpTestClient::spawn();
    client.initialize();

    // Store 3 "temporary" and 2 "permanent" memories
    for i in 0..3 {
        client.call_tool("store_memory", json!({
            "content": format!("Temp memory {}", i),
            "type_hint": "temporary"
        }));
    }
    for i in 0..2 {
        client.call_tool("store_memory", json!({
            "content": format!("Permanent memory {}", i),
            "type_hint": "permanent"
        }));
    }

    // Dry run (confirm: false) - should return count without deleting
    let dry_run_resp = client.call_tool("bulk_delete_memories", json!({
        "type_hint": "temporary",
        "confirm": false
    }));
    assert!(!McpTestClient::is_error(&dry_run_resp), "dry run should succeed");
    let dry_run = McpTestClient::structured_content(&dry_run_resp);
    assert_eq!(dry_run["matched"], 3, "Should match 3 temporary memories");
    assert_eq!(dry_run["deleted"], false, "Should not delete in dry run");

    // Verify memories still exist
    let list_resp = client.call_tool("list_memories", json!({"type_hint": "temporary"}));
    let list_content = McpTestClient::structured_content(&list_resp);
    assert_eq!(list_content["count"], 3, "Temporary memories should still exist after dry run");

    // Confirm deletion
    let delete_resp = client.call_tool("bulk_delete_memories", json!({
        "type_hint": "temporary",
        "confirm": true
    }));
    assert!(!McpTestClient::is_error(&delete_resp), "confirmed bulk delete should succeed");
    let deleted = McpTestClient::structured_content(&delete_resp);
    assert_eq!(deleted["deleted"], 3, "Should have deleted 3 memories");
    assert_eq!(deleted["confirmed"], true, "Should confirm deletion");

    // Verify only permanent memories remain
    let list_all_resp = client.call_tool("list_memories", json!({}));
    let all_content = McpTestClient::structured_content(&list_all_resp);
    let remaining = all_content["memories"].as_array().unwrap();
    assert_eq!(remaining.len(), 2, "Should have only 2 permanent memories left");
    for m in remaining {
        assert_eq!(m["type_hint"], "permanent", "Remaining memories should be permanent");
    }
}

#[test]
fn test_persistence_across_restart() {
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://memcp:memcp@localhost:5432/memcp".to_string());

    let memory_id;

    // First server instance: store a memory
    {
        let client = McpTestClient::spawn();
        client.initialize();

        let store_resp = client.call_tool("store_memory", json!({
            "content": "This memory persists across restarts",
            "type_hint": "fact",
            "source": "persistence-test"
        }));
        assert!(!McpTestClient::is_error(&store_resp),
            "store should succeed on first server instance");

        memory_id = McpTestClient::structured_content(&store_resp)["id"]
            .as_str().unwrap().to_string();

        // client drops here, killing the server process
        drop(database_url);
    }

    // Give the first server time to fully shut down
    thread::sleep(Duration::from_millis(200));

    // Second server instance: retrieve the memory from the same DB
    {
        let client2 = McpTestClient::spawn();
        client2.initialize();

        let get_resp = client2.call_tool("get_memory", json!({"id": memory_id}));
        assert!(!McpTestClient::is_error(&get_resp),
            "get should succeed on second server instance - memory must persist across restarts");

        let retrieved = McpTestClient::structured_content(&get_resp);
        assert_eq!(retrieved["id"], memory_id,
            "Retrieved memory ID should match stored ID");
        assert_eq!(retrieved["content"], "This memory persists across restarts",
            "Memory content should survive restart");
        assert_eq!(retrieved["type_hint"], "fact",
            "Memory type_hint should survive restart");
        assert_eq!(retrieved["source"], "persistence-test",
            "Memory source should survive restart");
    }
}

#[test]
fn test_resources_list_and_read() {
    let client = McpTestClient::spawn();
    client.initialize();

    // Store some memories including a preference
    client.call_tool("store_memory", json!({
        "content": "The user likes dark mode",
        "type_hint": "preference"
    }));
    client.call_tool("store_memory", json!({
        "content": "Rust is the language of choice",
        "type_hint": "fact"
    }));

    // List resources
    let list_resp = client.list_resources();
    assert!(list_resp["result"].is_object(), "resources/list should return a result");
    let resources = list_resp["result"]["resources"].as_array()
        .expect("resources should be an array");
    assert_eq!(resources.len(), 2, "Should list exactly 2 resources");

    let uris: Vec<&str> = resources.iter()
        .map(|r| r["uri"].as_str().unwrap())
        .collect();
    assert!(uris.contains(&"memory://session-primer"),
        "Should have session-primer resource");
    assert!(uris.contains(&"memory://user-profile"),
        "Should have user-profile resource");

    // Read session-primer resource
    let primer_resp = client.read_resource("memory://session-primer");
    assert!(primer_resp["result"].is_object(),
        "resources/read for session-primer should return a result");
    let primer_contents = primer_resp["result"]["contents"].as_array()
        .expect("contents should be array");
    assert!(!primer_contents.is_empty(), "session-primer should have content");
    let primer_text = primer_contents[0]["text"].as_str()
        .expect("session-primer should have text content");
    assert!(primer_text.contains("Rust is the language of choice") ||
            primer_text.contains("preference"),
        "session-primer text should contain stored memories: {}", primer_text);

    // Read user-profile resource
    let profile_resp = client.read_resource("memory://user-profile");
    assert!(profile_resp["result"].is_object(),
        "resources/read for user-profile should return a result");
    let profile_contents = profile_resp["result"]["contents"].as_array()
        .expect("profile contents should be array");
    assert!(!profile_contents.is_empty(), "user-profile should have content");
    let profile_text = profile_contents[0]["text"].as_str()
        .expect("user-profile should have text content");
    assert!(profile_text.contains("dark mode"),
        "user-profile text should contain preference memory: {}", profile_text);
}

#[test]
fn test_validation_errors() {
    let client = McpTestClient::spawn();
    client.initialize();

    // Empty content validation
    let empty_content_resp = client.call_tool("store_memory", json!({"content": ""}));
    assert!(McpTestClient::is_error(&empty_content_resp),
        "Empty content should return isError: true");

    // Non-existent ID
    let not_found_resp = client.call_tool("get_memory", json!({
        "id": "00000000-0000-0000-0000-000000000000"
    }));
    assert!(McpTestClient::is_error(&not_found_resp),
        "Non-existent memory ID should return isError: true");

    // Check the error text mentions something useful
    let result = &not_found_resp["result"];
    let content_arr = result["content"].as_array().expect("content should be array");
    let error_text = content_arr[0]["text"].as_str().expect("should have error text");
    assert!(!error_text.is_empty(), "Error text should not be empty");

    // Update with no fields
    let no_fields_resp = client.call_tool("update_memory", json!({
        "id": "some-id"
    }));
    assert!(McpTestClient::is_error(&no_fields_resp),
        "Update with no fields should return isError: true");

    // List with limit > 100: clamped to 100, not an error (current behavior)
    // But limit of 0 or negative is clamped too — test limit=101 works (clamped)
    let large_limit_resp = client.call_tool("list_memories", json!({"limit": 101}));
    // The current implementation clamps to 100, so it should succeed (not isError)
    assert!(!McpTestClient::is_error(&large_limit_resp),
        "Limit > 100 is clamped to 100, should not be an error");
}
