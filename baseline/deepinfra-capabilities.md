DeepInfra 模型：`zai-org/GLM-5.3-Flash`。
OpenAI SDK base URL：`https://api.deepinfra.com/v1/openai`。
密钥保存在本 worktree 的 `.env`，变量名 `DEEPINFRA_API_KEY`；该文件被 Git 忽略。

| 能力 | 官方文档 | 实际请求结果 |
| --- | --- | --- |
| Responses API | 文档目录未列出；公开 OpenAPI 没有 responses 路由 | SDK 请求 `/v1/openai/responses` 返回 404；`/v1/responses` 同样返回 404，body 均为 `{"detail":"Not Found"}` |
| Structured output | 支持 `response_format.type=json_schema` 和 `strict=true` | `client.chat.completions.parse(..., response_format=Extraction)` 返回 HTTP 200，Pydantic 解析得到 `{"name":"River","count":7,"tags":["blue","quiet"]}` |
| Tool calling | 支持 OpenAI-compatible function tools 和工具结果回传 | 返回 `add_integers(a=137,b=286)`；本地实际计算得到 423，使用对应 `tool_call_id` 回传后，模型返回 `{"result":423,"source":"add_integers"}` |

工具回传后的最后一轮同时使用 tools 和严格 JSON Schema，并通过 SDK structured-parse。
检查了实际 HTTP 请求、响应、工具参数、执行结果与后续消息，未设置 temperature、top_p 或输入/输出 token 上限。

原始证据位于 `runs/deepinfra-glm-5.3-flash-capabilities/`：
`01` 为 Responses 请求，`02` 为结构化抽取，`03` 为工具调用，`04` 为工具结果回传和结构化最终答案，`05` 为另一 Responses 路径。
每个编号各保存 request 和 response JSON；请求文件不包含认证头。`openapi-endpoints.json` 保存公开路由查询结果。
执行日志：`runs/deepinfra-capability-test.log`。

官方来源：

- [模型 API 文档](https://deepinfra.com/zai-org/GLM-5.3-Flash/api)
- [Structured Outputs](https://docs.deepinfra.com/chat/structured-outputs)
- [Tool Calling](https://docs.deepinfra.com/chat/tool-calling)
- [文档目录](https://docs.deepinfra.com/llms.txt)
- [公开 OpenAPI](https://api.deepinfra.com/openapi.json)
