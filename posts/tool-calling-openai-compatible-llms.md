---
title: "Tool calling with OpenAI-compatible LLMs: a complete reference"
date: 2026-08-18
tags: [llm, tool-calling, function-calling, openai-compatible, ai]
canonical: https://github.com/promptcracka/flippy
---

# Tool calling with OpenAI-compatible LLMs: a complete reference

"Tool calling" (or "function calling") is the feature that turns an LLM from
a text generator into an *agent*. Once the model can ask for a function
call, you can give it a calculator, a database, a web browser, or any
tool you want — and it can decide on its own when to use them.

If you've used the OpenAI API, you know the shape. The trick — and it's
the source of most of the bugs I see — is that **every provider implements
it slightly differently**, even though they all claim to be
"OpenAI-compatible."

## The shape

A tool-calling request looks like this:

```json
{
  "model": "...",
  "messages": [{"role": "user", "content": "..."}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather for a city.",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  }]
}
```

The model responds either with a normal `assistant` message or with a
`tool_calls` array, each entry looking like:

```json
{
  "id": "call_abc123",
  "type": "function",
  "function": {
    "name": "get_weather",
    "arguments": "{\"city\": \"Lagos\"}"
  }
}
```

You run the function, send the result back as a `role: tool` message with
the matching `tool_call_id`, and the model continues.

## The five things that bite people

### 1. Description matters more than the schema

The model decides whether to call your tool based mostly on the *description*.
"Get the current weather for a city" works. "Weather" does not. Spend time
on descriptions.

### 2. Arguments come back as a string, not a JSON object

`tool.function.arguments` is a stringified JSON. You must `json.loads()` it
yourself. This trips up almost every first implementation.

### 3. Required fields must actually be required

If your schema says `required: ["city"]` and the model thinks it knows the
city from context, it will sometimes omit it anyway. Mark required strictly,
and validate on your side.

### 4. Multi-turn is harder than single-turn

The model has to remember the function call result across turns. Some
providers handle this fine, some silently drop tool results past a certain
context length. Always include the full prior tool-call + tool-response
sequence in your messages list, never summarize it.

### 5. The model can hallucinate tool names

If you give the model three tools and it generates a fourth, that's your
problem to validate. Check `tool.function.name` against your registry
before calling anything.

## A working minimal example

The flippy project ships a 30-line tool-calling demo at the top of
[`src/aihub.py`](https://github.com/promptcracka/flippy). Run it with:

```bash
python src/aihub.py --tooltest
```

The model will respond with a `get_weather({"city": "Lagos"})` call. No
real weather service involved — just the call, so you can see the wire
shape and pattern-match it onto your own tools.

## When to skip tool calling

If your tool is just a calculator or a single database query, prompt
engineering is usually faster than wiring up tool calling. Reserve tool
calling for tools where the model genuinely needs to *decide* whether to
use them — a calculator the user explicitly invokes doesn't need it.
