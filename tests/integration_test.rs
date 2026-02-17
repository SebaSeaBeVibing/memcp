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

    // Give SQLite store a moment to initialize before the first request
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
