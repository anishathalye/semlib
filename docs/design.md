---
description: Learn about Semlib's design principles, type system, async patterns, and structured output handling for LLM-powered data processing.
---

# Library Design

Semlib makes use of [type annotations](https://typing.python.org/en/latest/spec/index.html), [generics](https://typing.python.org/en/latest/spec/generics.html), and [overloads](https://typing.python.org/en/latest/spec/overload.html). Semlib does not perform runtime checks for invalid arguments that would be caught by a type checker, so you should use a type checker like [mypy](https://mypy-lang.org/) for code utilizing Semlib.

Semlib follows a design pattern where many functions and methods (e.g., [map][semlib.session.Session.map]) accept a `return_type` argument that simplifies working with [structured output](https://platform.openai.com/docs/guides/structured-outputs). When `return_type` is not provided, the desired output is assumed to be a string. When `return_type` is provided, Semlib will parse the LLM's output into the specified type. Semlib uses [Pydantic](https://pydantic.dev/) for parsing and validation, so you can pass a Pydantic model as the `return_type`. As an alternative, Semlib provides the [Bare][semlib.bare.Bare] class, which can be used to mark a bare type like `int` or `list[float]` so that it can be used as the `return_type`.

Semlib uses [LiteLLM](https://github.com/BerriAI/litellm) as an abstraction layer for LLMs. You can use any LLM [supported by LiteLLM](https://docs.litellm.ai/docs/providers) with Semlib, choosing your model using the `model` argument. For example, you can set the `ANTHROPIC_API_KEY` environment variable and use `model="anthropic/claude-sonnet-4-20250514"`, or you can set up [Ollama](https://ollama.com/) and use `model="ollama_chat/gpt-oss:20b"`. Note that not all models support structured output, so if you want to use the `return_type` feature, you should choose a model that supports structured output. Semlib currently uses GPT-4o as the default model (and thus requires `OPENAI_API_KEY` to be set).

Semlib is optimized to be used in asynchronous code. Semlib also exports a synchronous interface (functions with the `_sync` suffix, which just wrap the asynchronous interface using [asyncio.run](https://docs.python.org/3/library/asyncio-runner.html#asyncio.run)). If for some reason you want to use the synchronous interface within asynchronous code (not recommended), you can consider [nest_asyncio](https://github.com/erdewit/nest_asyncio).
