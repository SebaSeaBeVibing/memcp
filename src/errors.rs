/// Domain-specific error types for memcp
///
/// Provides actionable error messages with detailed context to enable
/// AI agents to self-correct on bad tool calls.

#[derive(Debug, thiserror::Error)]
pub enum MemcpError {
    #[error("Validation error: {message}")]
    Validation {
        message: String,
        field: Option<String>
    },

    #[error("Memory not found: {id}")]
    NotFound {
        id: String
    },

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Storage error: {0}")]
    Storage(String),
}

impl From<sqlx::Error> for MemcpError {
    fn from(e: sqlx::Error) -> Self {
        MemcpError::Storage(e.to_string())
    }
}

impl From<crate::embedding::EmbeddingError> for MemcpError {
    fn from(e: crate::embedding::EmbeddingError) -> Self {
        MemcpError::Internal(e.to_string())
    }
}

impl MemcpError {
    /// Helper to create validation errors with field names
    ///
    /// Example:
    /// ```
    /// use memcp::errors::MemcpError;
    /// let err = MemcpError::validation("content", "Content cannot be empty");
    /// ```
    pub fn validation(field: &str, message: &str) -> Self {
        MemcpError::Validation {
            message: message.to_string(),
            field: Some(field.to_string()),
        }
    }
}
