/// LLM prompt templates for the LongMemEval benchmark pipeline.
///
/// Provides prompts for answer generation (with memory context + timestamps)
/// and GPT-4o judge evaluation (standard and abstention variants).

use crate::store::Memory;

/// Build the answer generation prompt. Includes retrieved memories with timestamps
/// for temporal reasoning support.
pub fn build_answer_prompt(
    question: &str,
    question_date: &str,
    retrieved_memories: &[Memory],
) -> String {
    let context = retrieved_memories
        .iter()
        .enumerate()
        .map(|(i, m)| {
            format!(
                "[Memory {}] (created: {})\n{}",
                i + 1,
                m.created_at.format("%Y-%m-%d"),
                m.content
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    format!(
        "You are a helpful assistant with access to a user's conversation history stored as memories.\n\n\
         Today's date: {question_date}\n\n\
         Relevant memories:\n{context}\n\n\
         Question: {question}\n\n\
         Answer the question based ONLY on the information in the memories above. \
         If the information is not available in the memories, say \"I don't have that information in my memory.\" \
         Be concise and direct."
    )
}

/// Build the standard judge prompt for non-abstention questions.
/// GPT-4o evaluates whether the hypothesis correctly answers the question given ground truth.
pub fn build_judge_prompt(question: &str, ground_truth: &str, hypothesis: &str) -> String {
    format!(
        "You are evaluating whether a chat assistant correctly answered a question based on its conversation memory.\n\n\
         Question: {question}\n\n\
         Ground truth answer: {ground_truth}\n\n\
         Assistant's response: {hypothesis}\n\n\
         Does the assistant's response correctly contain the ground truth answer? \
         The response doesn't need to match word-for-word, but must convey the same factual information. \
         Answer with only 'yes' or 'no'."
    )
}

/// Build the abstention judge prompt. Checks whether the model correctly identified
/// that it cannot answer the question (the question has a false premise or asks about
/// information not in the conversation history).
pub fn build_abstention_judge_prompt(question: &str, hypothesis: &str) -> String {
    format!(
        "You are evaluating whether a chat assistant correctly identified that a question \
         cannot be answered from its conversation history. The question has a false premise \
         or asks about information not in the history.\n\n\
         Question: {question}\n\n\
         Assistant's response: {hypothesis}\n\n\
         Did the assistant appropriately indicate it cannot answer, express uncertainty, \
         or decline to provide a specific answer? Answer with only 'yes' or 'no'."
    )
}
