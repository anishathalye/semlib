---
title: "Semlib: Semantic Data Processing"
description: A Python library for building data processing and analysis pipelines with LLMs.
---

# Semlib

Semlib is a Python library for building data processing and data analysis pipelines that leverage the power of large language models (LLMs). Semlib provides, as building blocks, familiar functional programming primitives like [map][semlib.session.Session.map], [reduce][semlib.session.Session.reduce], [sort][semlib.session.Session.sort], and [filter][semlib.session.Session.filter], but with a twist: Semlib's implementation of these operations are **programmed with natural language descriptions** rather than code. Under the hood, Semlib handles complexities such as prompting, parsing, concurrency control, caching, and cost tracking.

```pycon
>>> presidents = await prompt(
...     "Who were the 39th through 42nd presidents of the United States?",
...     return_type=Bare(list[str])
... )

>>> await sort(presidents, by="right-leaning", reverse=True)  # highest first
['Ronald Reagan', 'George H. W. Bush', 'Bill Clinton', 'Jimmy Carter']

>>> await find(presidents, by="former actor")
'Ronald Reagan'

>>> await map(
...     presidents,
...     "How old was {} when he took office?",
...     return_type=Bare(int),
... )
[52, 69, 64, 46]
```

## Rationale

Large language models are great at natural-language data processing and data analysis tasks, but when you have a large amount of data, you can't get high-quality results by just dumping all the data into a long-context LLM and asking it to complete a complex task in a single shot. Even with today's reasoning models and agents, this approach doesn't give great results.

This library provides an alternative. You can structure your computation using the building blocks that Semlib provides: functional programming primitives upgraded to handle semantic operations. This approach has a number of benefits.

**Quality.** By breaking down a sophisticated data processing task into simpler steps that are solved by today's LLMs, you can get higher-quality results, even in situations where today's LLMs might be capable of processing the data in a single shot and ending up with barely acceptable results. (example: analyzing support tickets in [Airline Support Report](./examples/airline-support/index.ipynb))

**Feasibility.** Even long-context LLMs have limitations (e.g., 1M tokens in today's frontier models). Furthermore, performance often drops off with longer inputs. By breaking down the data processing task into smaller steps, you can handle arbitrary-sized data. (example: sorting an arbitrary number of arXiv papers in [arXiv Paper Recommendations](./examples/arxiv-recommendations/index.ipynb))

**Latency.** By breaking down the computation into smaller pieces and structuring it using functional programming primitives like `map` and `reduce`, the parts of the computation can be run concurrently, reducing the latency of the overall computation.
 (example: tree [reduce][semlib.Session.reduce] with O(log n) computation depth in [Disneyland Reviews Synthesis](./examples/disneyland-reviews/index.ipynb))

**Cost.** By breaking down the computation into simpler sub-tasks, you can use smaller and cheaper models that are capable of solving those sub-tasks, which can reduce data processing costs. Furthermore, you can choose the model on a per-subtask basis, allowing you to further optimize costs. (example: using `gpt-4.1-nano` for the pre-filtering step in [arXiv Paper Recommendations](./examples/arxiv-recommendations/index.ipynb))

**Security.** By breaking down the computation into tasks that simpler models can handle, you can use open models that you host yourself, allowing you to process sensitive data without having to trust a third party. (example: using `gpt-oss` and `qwen3` in [Resume Filtering](./examples/resume-filtering/index.ipynb))

**Flexibility.** LLMs are great at certain tasks, like natural-language processing. They're not so great at other tasks, like multiplying numbers. Using Semlib, you can break down your data processing task into multiple steps, some of which use LLMs and others that just use regular old Python code, getting the best of both worlds. (example: Python code for filtering in [Resume Filtering](./examples/resume-filtering/index.ipynb))

Read more about the rationale, the story behind this library, and related work in the [**blog post**](https://anishathalye.com/semlib/).

## Citation

```bibtex
@misc{athalye:semlib,
  author = {Anish Athalye},
  title = {{Semlib}: Semantic data processing for {Python}},
  year = {2025},
  howpublished = {\url{https://github.com/anishathalye/semlib}},
}
```
