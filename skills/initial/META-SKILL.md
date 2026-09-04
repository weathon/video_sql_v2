# Reflection method

Re-enter the original conversation as the same answering agent. Review the actual sequence of hypotheses, tool inputs, tool outputs, evidence, answer, reference answer, and score.

1. Separate retrieval failure, reasoning failure, evidence-quality failure, and answer-format failure. Test competing explanations with the retained trace or new retrieval before choosing one. **Reason:** The final score alone cannot identify the failed stage.
2. Explain why the successful or unsuccessful behavior generalizes beyond the current question. Remove names, dates, exact answers, and other dataset-specific details; a sanitized example may illustrate the pattern. **Reason:** Skills should transfer without leaking evaluation content.
3. Compare the proposed lesson with existing skills. Prefer modifying or replacing a nearby rule over appending a duplicate; identify rules that should be deleted when contradicted or obsolete. **Reason:** Unbounded accumulation creates conflicts and makes retrieval harder.
4. Produce two outputs: task reflection for future retrieval and meta-reflection for improving later reflection itself. **Reason:** Learning how to diagnose traces is distinct from learning how to answer questions.
5. Preserve uncertainty when the trace does not discriminate between causes and state what evidence would distinguish them. **Reason:** Unsupported certainty turns one ambiguous rollout into a harmful general rule.
